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

from elastic_actuation_arm.calibration.environment.experiment_configuration_handler import \
    create_calibration_experiment_configs, CalibrationExperimentConfig
from elastic_actuation_arm.entities.payload.payload import Payload
from erpy.interfaces.mujoco.environment import MJCEnvironmentConfig
from erpy.interfaces.mujoco.phenome import MJCMorphology, MJCRobot
from erpy.utils.colors import rgba_orange, rgba_gray


def parse_calibration_environment_observations(observations: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    return {'shoulder_q0': observations[0],
            'elbow_q0': observations[1],
            'high_to_low': observations[2],
            'timestep': observations[3],
            'shoulder_q': observations[4],
            'shoulder_q_vel': observations[5],
            'elbow_q': observations[6],
            'elbow_q_vel': observations[7],
            'shoulder_torque': observations[8],
            'elbow_torque': observations[9],
            }


class CalibrationEnvironment(composer.Task):
    def __init__(self, config: CalibrationEnvironmentConfig, morphology: MJCMorphology) -> None:
        self.config = config

        self._arena = self._configure_arena()
        self._model = self._configure_model()
        self._robot = self._configure_robot(morphology=morphology)
        self._payload = self._configure_payload()
        self._task_observables = self._configure_task_observables()

        self.set_timesteps(control_timestep=self.config.control_timestep,
                           physics_timestep=self.config.physics_timestep)

    @property
    def root_entity(self) -> composer.Entity:
        return self._arena

    @property
    def task_observables(self) -> Dict:
        return self._task_observables

    def _configure_arena(self) -> Arena:
        arena = Floor()
        arena.mjcf_model.worldbody.add('geom', name='table', type='box', size=[0.8, 0.9, 1.0],
                                       pos=[0.0, 0.0, 1.0], rgba=rgba_orange)
        arena.mjcf_model.worldbody.add('geom', name='support', type='box', size=[0.1, 0.14, 0.4],
                                       pos=[-0.04, 0.0, 2.0], rgba=rgba_gray)

        arena.mjcf_model.worldbody.add(
            'camera',
            name='front_camera',
            pos=[0.0, -2.0, 2.2],
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
        if self.config.experiment_configuration.payload:
            payload = Payload(mass=5.0,
                              size=[0.05, 0.05, 0.05])

            self._robot.attach(payload, attach_site=self._robot.mjcf_body.find('site', 'end_effector_site'))

            return payload

    def _configure_robot(self, morphology: MJCMorphology) -> MJCMorphology:
        self._arena.attach(morphology)
        morphology.after_attachment()
        return morphology

    def _configure_task_observables(self) -> Dict[str, Observable]:
        task_observables = dict()

        def q0_observable(physics: mjcf.Physics) -> np.array:
            return self.config.experiment_configuration.q0

        task_observables["q0"] = observable.Generic(q0_observable)

        def h2l_observable(physics: mjcf.Physics) -> float:
            return float(self.config.experiment_configuration.high_to_low)

        task_observables["high_to_low"] = observable.Generic(h2l_observable)

        def time_observable(physics: mjcf.Physics) -> float:
            return physics.time() + self.config.experiment_configuration.real_world_start_timestep

        task_observables["timestep"] = observable.Generic(time_observable)

        def joint_observables(physics: mjcf.Physics) -> np.ndarary:
            shoulder_joint = self._robot.mjcf_model.find('joint', 'shoulder')
            elbow_joint = self._robot.mjcf_model.find('joint', 'elbow')

            shoulder_physics = physics.bind(shoulder_joint)
            elbow_physics = physics.bind(elbow_joint)

            shoulder_angle = shoulder_physics.qpos[0]
            shoulder_vel = shoulder_physics.qvel[0]

            elbow_angle = elbow_physics.qpos[0]
            elbow_vel = elbow_physics.qvel[0]

            return np.array([shoulder_angle, shoulder_vel,
                             elbow_angle, elbow_vel])

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
        if self.config.experiment_configuration.payload:
            new_pos = np.array(self._payload.get_pose(physics)[0])
            initial_position = np.array(new_pos)
            initial_quaternion = euler2quat(0, 0, 0)

            self._payload.set_pose(physics=physics,
                                   position=initial_position,
                                   quaternion=initial_quaternion)

    def _configure_initial_pose(self, physics: mjcf.Physics) -> None:
        self._configure_initial_robot_pose(physics)
        self._configure_initial_payload_pose(physics)

    def _configure_initial_joint_values(self, physics: mjcf.Physics) -> None:
        shoulder_joint = self._robot.mjcf_model.find('joint', 'shoulder')
        shoulder_actuator = self._robot.mjcf_model.find('actuator', 'shoulder_p_actuator')
        elbow_joint = self._robot.mjcf_model.find('joint', 'elbow')
        elbow_actuator = self._robot.mjcf_model.find('actuator', 'elbow_p_actuator')

        shoulder_physics = physics.bind(shoulder_joint)
        elbow_physics = physics.bind(elbow_joint)

        shoulder_angle, elbow_angle = self.config.experiment_configuration.q0
        shoulder_physics.qpos[0] = shoulder_angle
        elbow_physics.qpos[0] = elbow_angle

        shoulder_vel, elbow_vel = self.config.experiment_configuration.q_vel0
        shoulder_physics.qvel[0] = shoulder_vel
        elbow_physics.qvel[0] = elbow_vel

    def initialize_episode(self, physics: mjcf.Physics, random_state) -> None:
        self._configure_initial_pose(physics=physics)
        self._configure_initial_joint_values(physics=physics)

    def get_reward(self, physics: Physics) -> float:
        # Reward calculation is not possible here -> environment should not calculate the NRMSE
        return 0

    def get_info(self, time_step, physics) -> Dict:
        info = {}
        return info


@dataclass
class CalibrationEnvironmentConfig(MJCEnvironmentConfig):
    real_world_data_path: str
    experiment_ids: List[int]
    _experiment_configurations: List[CalibrationExperimentConfig] = None
    contacts_enabled: bool = True
    experiment_index: int = 0

    @property
    def task(self) -> Callable[[MJCEnvironmentConfig, MJCRobot], composer.Task]:
        return CalibrationEnvironment

    @property
    def camera_ids(self) -> List[int]:
        return [1]

    @property
    def experiment_configurations(self) -> List[CalibrationExperimentConfig]:
        if self._experiment_configurations is None:
            self._experiment_configurations = create_calibration_experiment_configs(base_path=self.real_world_data_path,
                                                                                    experiment_ids=self.experiment_ids)
        return self._experiment_configurations

    @property
    def experiment_configuration(self) -> CalibrationExperimentConfig:
        return self.experiment_configurations[self.experiment_index]

    @property
    def simulation_time(self) -> float:
        return 58.0

    @property
    def num_substeps(self) -> int:
        return 3

    @property
    def time_scale(self) -> float:
        return 1.0

