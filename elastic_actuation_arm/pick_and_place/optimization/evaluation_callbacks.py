from pathlib import Path
from typing import cast, Tuple

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt

from elastic_actuation_arm.entities.manipulator.elastic_actuation import calculate_load_torque
from elastic_actuation_arm.pick_and_place.environment.environment import PickAndPlaceEnvironmentConfig, \
    parse_pickandplace_environment_observations
from elastic_actuation_arm.pick_and_place.optimization.robot.robot import ManipulatorPickAndPlaceSpringTrajectoryRobot
from erpy.base.ea import EAConfig
from erpy.base.evaluator import EvaluationCallback, EvaluationResult
from erpy.base.genome import Genome
from erpy.utils.colors import rgba_green


class AdaptEnvironmentConfigCallback(EvaluationCallback):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(name="AdaptEnvironmentConfigCallback", config=config)
        self._go_duration = None
        self._ret_duration = None

    def from_robot(self, robot: ManipulatorPickAndPlaceSpringTrajectoryRobot) -> None:
        self._go_duration = robot.controller.specification.go_duration.value
        self._ret_duration = robot.controller.specification.ret_duration.value

    def update_environment_config(self, environment_config: PickAndPlaceEnvironmentConfig) -> None:
        environment_config.go_duration = self._go_duration
        environment_config.ret_duration = self._ret_duration


class JointVelocityPenaltyCallback(EvaluationCallback):
    def __init__(self, config: EAConfig):
        super().__init__(config, name="JointVelocityLimitCallback")

        self._genome_id = None
        self._environment_config = cast(PickAndPlaceEnvironmentConfig, self.config.environment_config)
        self._simulation_measurements = []
        self._measurement_keys = ['shoulder_q_vel', 'elbow_q_vel']

        self._base_path = Path(self._ea_config.saver_config.analysis_path) / "trajectory_plots"
        self._base_path.mkdir(parents=True, exist_ok=True)

    def before_step(self, observations, actions):
        # Parse the observations -> get measurements
        obs_dict = parse_pickandplace_environment_observations(observations)

        if obs_dict['timestep'] > self._environment_config.initialisation_duration:
            relative_time = obs_dict['timestep'] - self._environment_config.initialisation_duration

            in_go_phase = relative_time < self._environment_config.go_duration
            ret_phase_start = self._environment_config.go_duration + self._environment_config.placement_duration
            in_ret_phase = relative_time > ret_phase_start

            if in_go_phase or in_ret_phase:
                measurement = [obs_dict[key] for key in self._measurement_keys]
                self._simulation_measurements.append(measurement)

    def after_episode(self):
        self._simulation_measurements = np.array(self._simulation_measurements)

    def from_genome(self, genome: Genome) -> None:
        self._genome_id = genome.genome_id

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        shoulder_q_vel = self._simulation_measurements[:, 0] / np.pi * 180
        elbow_q_vel = self._simulation_measurements[:, 1] / np.pi * 180

        shoulder_invalid = any(np.abs(shoulder_q_vel) >= 120)
        elbow_invalid = any(np.abs(elbow_q_vel) >= 120)
        if shoulder_invalid:
            evaluation_result.fitness = -5 * abs(evaluation_result.fitness)
        if elbow_invalid:
            evaluation_result.fitness = -5 * abs(evaluation_result.fitness)

        evaluation_result.info["JointVelocityLimitCallback"] = {"shoulder-invalid": shoulder_invalid,
                                                                "elbow-invalid": elbow_invalid}
        return evaluation_result


class QQVelTorqueRecorderCallback(EvaluationCallback):
    def __init__(self, config: EAConfig, name: str):
        super().__init__(config, name=name)

        self._environment_config = cast(PickAndPlaceEnvironmentConfig, self.config.environment_config)
        self._simulation_measurements = []
        self._measurement_keys = ['timestep',
                                  'shoulder_q', 'elbow_q',
                                  'shoulder_q_vel', 'elbow_q_vel',
                                  'shoulder_torque', 'elbow_torque']

    def after_episode(self):
        self._simulation_measurements = np.array(self._simulation_measurements)

    def before_step(self, observations, actions):
        # Parse the observations -> get measurements
        obs_dict = parse_pickandplace_environment_observations(observations)

        if obs_dict['timestep'] > self._environment_config.initialisation_duration:
            relative_time = obs_dict['timestep'] - self._environment_config.initialisation_duration

            in_go_phase = relative_time < self._environment_config.go_duration
            ret_phase_start = self._environment_config.go_duration + self._environment_config.placement_duration
            in_ret_phase = relative_time > ret_phase_start

            if in_go_phase or in_ret_phase:
                measurement = [in_go_phase] + [obs_dict[key] for key in self._measurement_keys]
                self._simulation_measurements.append(measurement)


class LTTrajectoryCallback(QQVelTorqueRecorderCallback):
    def __init__(self, config: EAConfig):
        super().__init__(config, name="LTTrajectoryCallback")
        self._base_path = Path(self._ea_config.saver_config.analysis_path) / "trajectory_plots"
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._genome_id = None

    def from_genome(self, genome: Genome) -> None:
        self._genome_id = genome.genome_id

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        # Remove time-step column
        timesteps = self._simulation_measurements[:, 0] - self._environment_config.initialisation_duration
        simulation_trajectory = self._simulation_measurements[:, 1:]

        # Add acceleration to simulation data
        shoulder_q_vel = simulation_trajectory[:, 2]
        elbow_q_vel = simulation_trajectory[:, 3]

        shoulder_q_acc = np.gradient(shoulder_q_vel, self._environment_config.control_timestep)
        elbow_q_acc = np.gradient(elbow_q_vel, self._environment_config.control_timestep)

        # Calculate the load torques
        shoulder_load_torque = calculate_load_torque(joint_index=1,
                                                     torque=simulation_trajectory[:, 4],
                                                     vel=shoulder_q_vel,
                                                     acc=shoulder_q_acc)
        elbow_load_torque = calculate_load_torque(joint_index=2,
                                                  torque=simulation_trajectory[:, 5],
                                                  vel=elbow_q_vel,
                                                  acc=elbow_q_acc)

        simulation_load_torques = np.vstack([shoulder_load_torque, elbow_load_torque])
        labels = ["shoulder", "elbow"]

        for label, load_torques in zip(labels, simulation_load_torques):
            plt.plot(timesteps, load_torques, label=label, color=rgba_green, linestyle='--')
            plt.ylabel('load torque')
            plt.xlabel('time (s)')

            t1 = self._environment_config.go_duration
            t2 = t1 + self._environment_config.placement_duration
            t3 = t2 + self._environment_config.ret_duration
            plt.xticks([0, t1, t2, t3])
            path = str(self._base_path / f"genome_{self._genome_id}_plot_{label}_load_torque.png")
            plt.savefig(path)
            plt.close()

        return evaluation_result


class TrajectorySaverCallback(QQVelTorqueRecorderCallback):
    def __init__(self, config: EAConfig):
        super().__init__(config, name="TrajectorySaverCallback")

        self._base_path = Path(self._ea_config.saver_config.analysis_path) / "trajectory_data"
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._genome_id = None

    def from_genome(self, genome: Genome) -> None:
        self._genome_id = genome.genome_id

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        # Split into go and return phase
        go_measurements = self._simulation_measurements[self._simulation_measurements[:, 0] == 1][:, 1:]
        return_measurements = self._simulation_measurements[self._simulation_measurements[:, 0] == 0][:, 1:]

        genome_id = self._genome_id
        control_timestep = self._environment_config.control_timestep

        def create_df(name: str, measurements: np.ndarray) -> None:
            # Align real world timesteps with simulation timesteps
            #   Make simulation timesteps start from 0
            measurements[:, 0] -= measurements[0, 0]

            timesteps = measurements[:, 0]
            shoulder_q = measurements[:, 1]
            elbow_q = measurements[:, 2]
            shoulder_q_vel = measurements[:, 3]
            elbow_q_vel = measurements[:, 4]
            shoulder_q_acc = np.gradient(shoulder_q_vel, control_timestep)
            elbow_q_acc = np.gradient(elbow_q_vel, control_timestep)
            shoulder_torque = measurements[:, 5]
            elbow_torque = measurements[:, 6]

            COLUMNS = ["time",
                       "q", "q_vel", "q_acc",
                       "useless1", "torque", "useless2"]
            # Shoulder data
            shoulder_data = np.vstack((timesteps,
                                       shoulder_q, shoulder_q_vel, shoulder_q_acc,
                                       np.zeros(timesteps.shape), shoulder_torque, np.zeros(timesteps.shape)
                                       )).T
            shoulder_df = pd.DataFrame(data=shoulder_data, columns=COLUMNS)

            # Elbow data
            elbow_data = np.vstack((timesteps,
                                    elbow_q, elbow_q_vel, elbow_q_acc,
                                    np.zeros(timesteps.shape), elbow_torque, np.zeros(timesteps.shape))).T
            elbow_df = pd.DataFrame(data=elbow_data, columns=COLUMNS)

            # Save dfs
            shoulder_path = str(self._base_path / f"genome_{genome_id}_{name}_shoulder.csv")
            elbow_path = str(self._base_path / f"genome_{genome_id}_{name}_elbow.csv")

            shoulder_df.to_csv(path_or_buf=shoulder_path, sep=' ', index=False, header=False)
            elbow_df.to_csv(path_or_buf=elbow_path, sep=' ', index=False, header=False)

        create_df(name="go", measurements=go_measurements)
        create_df(name="return", measurements=return_measurements)

        return evaluation_result


class LoadTorqueIntegralSquaredFitnessCallback(QQVelTorqueRecorderCallback):
    def __init__(self, config: EAConfig):
        super().__init__(config, name="LoadTorqueIntegralSquaredFitnessCallback")

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        # Split into go and return phase
        go_measurements = self._simulation_measurements[self._simulation_measurements[:, 0] == 1][:, 1:]
        return_measurements = self._simulation_measurements[self._simulation_measurements[:, 0] == 0][:, 1:]

        def calculate_lts(measurements: np.ndarray) -> Tuple[float, float]:
            # Align real world timesteps with simulation timesteps
            #   Make simulation timesteps start from 0
            measurements[:, 0] -= measurements[0, 0]

            timesteps = measurements[:, 0]
            shoulder_q_vel = measurements[:, 3]
            elbow_q_vel = measurements[:, 4]
            shoulder_q_acc = np.gradient(shoulder_q_vel, timesteps)
            elbow_q_acc = np.gradient(elbow_q_vel, timesteps)
            shoulder_torque = measurements[:, 5]
            elbow_torque = measurements[:, 6]

            shoulder_load_torque = calculate_load_torque(joint_index=1,
                                                         torque=shoulder_torque,
                                                         vel=shoulder_q_vel,
                                                         acc=shoulder_q_acc)
            elbow_load_torque = calculate_load_torque(joint_index=2,
                                                      torque=elbow_torque,
                                                      vel=elbow_q_vel,
                                                      acc=elbow_q_acc)
            shoulder_load_torque = np.power(shoulder_load_torque, 2)
            elbow_load_torque = np.power(elbow_load_torque, 2)
            shoulder_lts = scipy.integrate.simpson(y=shoulder_load_torque,
                                                   x=timesteps)
            elbow_lts = scipy.integrate.simpson(y=elbow_load_torque,
                                                x=timesteps)

            return shoulder_lts, elbow_lts

        go_shoulder_lts, go_elbow_lts = calculate_lts(go_measurements)
        ret_shoulder_lts, ret_elbow_lts = calculate_lts(return_measurements)
        shoulder_lts = go_shoulder_lts + ret_shoulder_lts
        elbow_lts = go_elbow_lts + ret_elbow_lts

        # ERPY always maximises fitness -> negate total load torque
        evaluation_result.fitness = -(shoulder_lts + elbow_lts)
        evaluation_result.info["logging_lt_integral_shoulder"] = shoulder_lts
        evaluation_result.info["logging_lt_integral_elbow"] = elbow_lts
        evaluation_result.info["logging_lt_integral_total"] = shoulder_lts + elbow_lts

        return evaluation_result
