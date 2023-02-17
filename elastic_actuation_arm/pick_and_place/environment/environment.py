from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable, List, Union

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer import Arena
from dm_control.composer.observation import observable
from dm_control.composer.observation.observable import Observable
from dm_control.locomotion.arenas import Floor
from dm_control.mjcf import Physics
from dm_control.mujoco.math import euler2quat

from elastic_actuation_arm.entities.manipulator.manipulator import ManipulatorMorphology
from elastic_actuation_arm.entities.payload.payload import Payload
from erpy.interfaces.mujoco.environment import MJCEnvironmentConfig
from erpy.interfaces.mujoco.phenome import MJCRobot
from erpy.utils.colors import rgba_gray


def parse_pickandplace_environment_observations(observations: np.ndarray) -> Dict[
    str, Union[float, np.ndarray]]:
    labels = ['timestep', 'initialisation_duration', 'go_duration', 'placement_duration', 'ret_duration',
              'shoulder_q', 'elbow_q',
              'shoulder_q_vel', 'elbow_q_vel',
              'shoulder_torque', 'elbow_torque']
    return {label: value for label, value in zip(labels, observations)}


class PickAndPlaceEnvironment(composer.Task):
    def __init__(self, config: PickAndPlaceEnvironmentConfig, morphology: ManipulatorMorphology) -> None:
        self.config = config

        self._arena = self._configure_arena()
        self._model = self._configure_model()
        self._robot = self._configure_robot(morphology=morphology)
        self._payload = self._configure_payload()
        self._task_observables = self._configure_task_observables()

        self.set_timesteps(control_timestep=self.config.control_timestep,
                           physics_timestep=self.config.physics_timestep)

        self._payload_in_use = True

    @property
    def root_entity(self) -> composer.Entity:
        return self._arena

    @property
    def task_observables(self) -> Dict:
        return self._task_observables

    def _configure_arena(self) -> Arena:
        try:
            arena = Floor(aesthetic='white')
        except Exception:
            arena = Floor()
        arena.mjcf_model.worldbody.add('geom', name='table', type='box', size=[0.8, 0.9, 1.0],
                                       pos=[0.0, 0.0, 1.0], rgba=[0, 0, 0, 0])
        arena.mjcf_model.worldbody.add('geom', name='support', type='box', size=[0.1, 0.14, 0.4],
                                       pos=[-0.04, 0.0, 2.0], rgba=rgba_gray)
        arena.mjcf_model.worldbody.add(
            'camera',
            name='front_camera',
            pos=[0.3, -1.2, 2.5],
            quat=euler2quat(90, 0, 0)
        )

        return arena

    def _configure_model(self) -> mjcf.RootElement:
        model = self._arena.mjcf_model
        model.compiler.fusestatic = True

        if self.config.contacts_enabled:
            model.option.flag.filterparent = 'disable'
        else:
            model.option.flag.contact = 'disable'

        getattr(model.visual, "global").offwidth = 1000
        getattr(model.visual, "global").offheight = 1000

        return model

    def _configure_payload(self) -> Payload:
        payload = Payload(mass=5.0,
                          size=[0.05, 0.05, 0.05])

        attachment_frame = self._arena.attach(payload)
        h_slide = attachment_frame.add('joint', type='slide', axis=[1, 0, 0], name='payload_horizontal')
        attachment_frame.add('joint', type='slide', axis=[0, 0, 1], name='payload_vertical')
        attachment_frame.add('joint', type='hinge', axis=[0, 1, 0], name='payload_hinge')

        # Make sure payload doesn't collide with forearm
        self._model.contact.add('exclude', body1=payload.mjcf_body,
                                body2=self._robot.fore_arm.mjcf_body)

        # Add an actuator to the payload so that it can be moved
        self._model.actuator.add('motor', name='payload_actuator',
                                 joint=h_slide,
                                 ctrllimited=True,
                                 ctrlrange=[0.0, 1.0],
                                 forcelimited=True,
                                 gear=[100],
                                 forcerange=[0.0, 1000])

        return payload

    def _configure_robot(self, morphology: ManipulatorMorphology) -> ManipulatorMorphology:
        self._arena.attach(morphology)
        morphology.after_attachment()
        return morphology

    def _configure_task_observables(self) -> Dict[str, Observable]:
        task_observables = dict()

        def time_observable(physics: mjcf.Physics) -> np.ndarray:
            time = physics.time()
            initialisation_duration = self.config.initialisation_duration
            go_duration = self.config.go_duration
            placement_duration = self.config.placement_duration
            ret_duration = self.config.ret_duration
            return np.array([time, initialisation_duration, go_duration, placement_duration, ret_duration])

        task_observables["times"] = observable.Generic(time_observable)

        def joint_observables(physics: mjcf.Physics) -> np.ndarary:
            shoulder_joint = self._robot.mjcf_model.find('joint', 'shoulder')
            elbow_joint = self._robot.mjcf_model.find('joint', 'elbow')

            shoulder_physics = physics.bind(shoulder_joint)
            elbow_physics = physics.bind(elbow_joint)

            shoulder_angle = shoulder_physics.qpos[0]
            elbow_angle = elbow_physics.qpos[0]

            shoulder_vel = shoulder_physics.qvel[0]
            elbow_vel = elbow_physics.qvel[0]

            return np.array([shoulder_angle, elbow_angle,
                             shoulder_vel, elbow_vel])

        task_observables["joint_observables"] = observable.Generic(joint_observables)

        def actuator_observables(physics: mjcf.Physics) -> np.ndarray:
            shoulder_torque_sensor = self._robot.mjcf_model.find('sensor', 'shoulder_actuator_torque')
            elbow_torque_sensor = self._robot.mjcf_model.find('sensor', 'elbow_actuator_torque')

            shoulder_torque = physics.bind(shoulder_torque_sensor).sensordata[0]
            elbow_torque = physics.bind(elbow_torque_sensor).sensordata[0]

            return np.array([shoulder_torque, elbow_torque])

        task_observables["actuator_observables"] = observable.Generic(actuator_observables)

        for t_obs in task_observables.values():
            t_obs.enabled = True

        return task_observables

    def _configure_initial_robot_pose(self, physics: mjcf.Physics) -> None:
        initial_position = np.array([0.0, 0.0, 2.4])
        initial_quaternion = euler2quat(0, 0, 0)

        self._robot.set_pose(physics=physics,
                             position=initial_position,
                             quaternion=initial_quaternion)

    def _configure_initial_payload_pose(self, physics: mjcf.Physics) -> None:
        initial_position = np.array([0.615, 0.0, 2.3 + self._payload.size[2] + 0.125])
        initial_quaternion = euler2quat(0, 0, 0)

        self._payload.set_pose(physics=physics,
                               position=initial_position,
                               quaternion=initial_quaternion)

    def _configure_initial_pose(self, physics: mjcf.Physics) -> None:
        self._configure_initial_robot_pose(physics)
        self._configure_initial_payload_pose(physics)

    def initialize_episode(self, physics: mjcf.Physics, random_state) -> None:
        self._configure_initial_pose(physics=physics)
        self._payload_in_use = True

    def get_reward(self, physics: Physics) -> float:
        return 0.0

    def get_info(self, time_step, physics) -> Dict:
        info = {}
        return info


@dataclass
class PickAndPlaceEnvironmentConfig(MJCEnvironmentConfig):
    contacts_enabled: bool = True
    initialisation_duration: float = 4.0
    go_duration: float = 3.0
    placement_duration: float = 2.0
    ret_duration: float = 3.0

    @property
    def task(self) -> Callable[[MJCEnvironmentConfig, MJCRobot], composer.Task]:
        return PickAndPlaceEnvironment

    @property
    def camera_ids(self) -> List[int]:
        return [1]

    @property
    def simulation_time(self) -> float:
        return self.initialisation_duration + self.go_duration + self.placement_duration + self.ret_duration

    @property
    def num_substeps(self) -> int:
        return 3

    @property
    def time_scale(self) -> float:
        return 1.0
