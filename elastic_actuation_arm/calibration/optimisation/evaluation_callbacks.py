from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from elastic_actuation_arm.calibration.environment.environment import CalibrationEnvironmentConfig, \
    parse_calibration_environment_observations
from elastic_actuation_arm.calibration.environment.experiment_configuration_handler import \
    align_measurements
from elastic_actuation_arm.calibration.optimisation.robot.genome import ManipulatorCalibrationGenome
from elastic_actuation_arm.entities.manipulator.elastic_actuation import calculate_load_torque
from erpy.base.ea import EAConfig
from erpy.base.evaluator import EvaluationCallback, EvaluationResult
from erpy.base.genome import Genome
from erpy.utils.colors import rgba_green, rgba_red


class UpdateExperimentIndexCallback(EvaluationCallback):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config=config, name="UpdateExperimentIndexCallback")
        self._environment_config = cast(CalibrationEnvironmentConfig, self.config.environment_config)
        self._experiment_index = 0

    def before_episode(self):
        self._environment_config.experiment_index = self._experiment_index

    def after_episode(self):
        self._experiment_index += 1


class SpringStiffnessPerturbationCallback(EvaluationCallback):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config=config, name="SpringStiffnessPerturbationCallback")

        self._environment_config = cast(CalibrationEnvironmentConfig, self.config.environment_config)
        self._experiment_configurations = self._environment_config.experiment_configurations
        self._original_spring_stiffness = np.array([])
        self._genome: ManipulatorCalibrationGenome = None

    def after_episode(self):
        # Reset to original spring stiffness values
        self._genome.specification.morphology_specification.upper_arm_spec.spring_spec.stiffness.value = \
            self._original_spring_stiffness[0]
        self._genome.specification.morphology_specification.fore_arm_spec.spring_spec.stiffness.value = \
            self._original_spring_stiffness[1]

    def from_genome(self, genome: ManipulatorCalibrationGenome):
        # Store initial spring constants
        self._genome = genome
        morph_spec = genome.specification.morphology_specification
        shoulder_stiffness = morph_spec.upper_arm_spec.spring_spec.stiffness.value
        elbow_stiffness = morph_spec.fore_arm_spec.spring_spec.stiffness.value

        self._original_spring_stiffness = np.array([shoulder_stiffness, elbow_stiffness])

    def before_episode(self) -> None:
        # Adapt spring stiffness based on current experiment configuration
        spring_stiffness_factor = self._environment_config.experiment_configuration.spring_stiffness_factor

        new_spring_stiffness = spring_stiffness_factor * self._original_spring_stiffness

        self._genome.specification.morphology_specification.upper_arm_spec.spring_spec.stiffness.value = \
            new_spring_stiffness[0]
        self._genome.specification.morphology_specification.fore_arm_spec.spring_spec.stiffness.value = \
            new_spring_stiffness[1]


class NRMSEFitnessCallback(EvaluationCallback):
    def __init__(self, config: EAConfig):
        super().__init__(config, name="NRMSEFitnessCallback")

        self._environment_config = cast(CalibrationEnvironmentConfig, self.config.environment_config)
        self._experiment_configurations = self._environment_config.experiment_configurations
        self._current_measurements = []
        self._all_measurements = []

        start_seconds = 1
        timesteps_per_second = self._environment_config.num_timesteps / self._environment_config.simulation_time
        self.start_timestep = int(start_seconds * timesteps_per_second)

    def after_episode(self):
        # Load the next experiment config
        self._all_measurements.append(np.array(self._current_measurements))
        self._current_measurements = []

    def before_step(self, observations, actions):
        # Parse the observations -> get measurements
        obs_dict = parse_calibration_environment_observations(observations)

        measurement_keys = ['timestep',
                            'shoulder_q', 'elbow_q',
                            'shoulder_qvel', 'elbow_qvel',
                            'shoulder_qacc', 'elbow_qacc',
                            'shoulder_torque', 'elbow_torque',
                            'shoulder_load_torque', 'elbow_load_torque']
        measurement = [obs_dict[key] for key in measurement_keys]

        self._current_measurements.append(measurement)

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        nrmse_per_trajectory = []

        for simulation_trajectory, experiment_config in zip(self._all_measurements, self._experiment_configurations):
            real_world_trajectory = experiment_config.measurements

            real_world_trajectory = align_measurements(values=real_world_trajectory,
                                                       simulation_timesteps=simulation_trajectory[:, 0],
                                                       real_world_timesteps=experiment_config.timestep)

            # Start after n seconds have passed
            simulation_trajectory = simulation_trajectory[self.start_timestep: -self.start_timestep, :]
            real_world_trajectory = real_world_trajectory[self.start_timestep: -self.start_timestep, :]

            real_world_observation_range = real_world_trajectory.max(axis=0) - real_world_trajectory.min(axis=0)

            simulation_trajectory = simulation_trajectory[:, 1:]  # drop the time-step column

            rmse = mean_squared_error(y_true=real_world_trajectory, y_pred=simulation_trajectory,
                                      squared=False, multioutput='raw_values')
            nrmse = rmse / real_world_observation_range

            nrmse_per_trajectory.append(nrmse)

        # Sum the NRMSE over the different dimensions, sum across trajectories
        total_error = np.sum(nrmse_per_trajectory)

        # ERPY always maximises fitness -> negate the NRMSE
        evaluation_result.fitness = -total_error

        # Add average NRMSE per dimension to evaluation_result.info for logging purposes
        evaluation_result.info["logging_nrmse"] = np.average(nrmse_per_trajectory, axis=0)

        return evaluation_result


class LoadTorqueNRMSEFitnessCallback(EvaluationCallback):
    def __init__(self, config: EAConfig):
        super().__init__(config, name="LoadTorqueNRMSEFitnessCallback")

        self._environment_config = cast(CalibrationEnvironmentConfig, self.config.environment_config)
        self._current_measurements = []
        self._all_measurements = []
        self._measurement_keys = ['shoulder_q', 'elbow_q',
                                  'shoulder_q_vel', 'elbow_q_vel',
                                  'shoulder_torque', 'elbow_torque']

    def after_episode(self):
        self._all_measurements.append(np.array(self._current_measurements))
        self._current_measurements = []

    def before_step(self, observations, actions):
        obs_dict = parse_calibration_environment_observations(observations)

        measurement_keys = ['timestep'] + self._measurement_keys
        measurement = [obs_dict[key] for key in measurement_keys]

        self._current_measurements.append(measurement)

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        nrmse_per_trajectory = []

        for simulation_trajectory, experiment_config in zip(self._all_measurements,
                                                            self._environment_config.experiment_configurations):
            real_world_trajectory = experiment_config.measurements

            # Align real world timesteps with simulation timesteps
            #   Make simulation timesteps start from 0
            simulation_trajectory[:, 0] -= simulation_trajectory[0, 0]

            # Remove time-step column
            timesteps = simulation_trajectory[:, 0]
            simulation_trajectory = simulation_trajectory[:, 1:]

            # Add acceleration to simulation data
            shoulder_q_vel = simulation_trajectory[:, 2]
            elbow_q_vel = simulation_trajectory[:, 3]

            shoulder_q_acc = np.gradient(shoulder_q_vel, self._environment_config.control_timestep)
            elbow_q_acc = np.gradient(elbow_q_vel, self._environment_config.control_timestep)
            simulation_acceleration = np.vstack([shoulder_q_acc, elbow_q_acc]).T

            # Calculate the load torques
            shoulder_load_torque = calculate_load_torque(joint_index=1,
                                                         torque=simulation_trajectory[:, 4],
                                                         vel=shoulder_q_vel,
                                                         acc=shoulder_q_acc)
            elbow_load_torque = calculate_load_torque(joint_index=2,
                                                      torque=simulation_trajectory[:, 5],
                                                      vel=elbow_q_vel,
                                                      acc=elbow_q_acc)

            simulation_load_torques = np.vstack([shoulder_load_torque, elbow_load_torque]).T

            simulation_trajectory = np.concatenate([simulation_trajectory,
                                                    simulation_acceleration,
                                                    simulation_load_torques], axis=1)

            #  Align the real world measurements with the simulation timesteps
            real_world_trajectory = align_measurements(values=real_world_trajectory,
                                                       simulation_timesteps=timesteps,
                                                       real_world_timesteps=experiment_config.timestep)

            real_world_observation_range = real_world_trajectory.max(axis=0) - real_world_trajectory.min(axis=0)
            rmse = mean_squared_error(y_true=real_world_trajectory, y_pred=simulation_trajectory,
                                      squared=False, multioutput='raw_values')
            nrmse = rmse / real_world_observation_range

            nrmse_per_trajectory.append(nrmse)

        nrmse_per_trajectory = np.array(nrmse_per_trajectory)
        # Sum the NRMSE over the different dimensions, sum across trajectories
        total_error = np.sum(nrmse_per_trajectory[:, -2:])

        # ERPY always maximises fitness -> negate the NRMSE
        evaluation_result.fitness = -total_error

        # Add average NRMSE per dimension to evaluation_result.info for logging purposes
        evaluation_result.info["logging_nrmse"] = np.average(nrmse_per_trajectory, axis=0)

        return evaluation_result


class TrajectoryPlotCallback(EvaluationCallback):
    def __init__(self, config: EAConfig):
        super().__init__(config, name="TrajectoryPlotCallback")

        self._genome_id = None
        self._environment_config = cast(CalibrationEnvironmentConfig, self.config.environment_config)
        self._experiment_configurations = self._environment_config.experiment_configurations
        self._current_measurements = []
        self._all_measurements = []
        self._measurement_keys = ['shoulder_q', 'elbow_q',
                                  'shoulder_q_vel', 'elbow_q_vel',
                                  'shoulder_torque', 'elbow_torque']

        self._base_path = Path(self._ea_config.saver_config.analysis_path) / "trajectory_plots"
        self._base_path.mkdir(parents=True, exist_ok=True)

    def after_episode(self):
        # Load the next experiment config
        self._all_measurements.append(np.array(self._current_measurements))
        self._current_measurements = []

    def before_step(self, observations, actions):
        # Parse the observations -> get measurements
        obs_dict = parse_calibration_environment_observations(observations)

        measurement_keys = ['timestep'] + self._measurement_keys
        measurement = [obs_dict[key] for key in measurement_keys]

        self._current_measurements.append(measurement)

    def from_genome(self, genome: Genome) -> None:
        self._genome_id = genome.genome_id

    def update_evaluation_result(self, evaluation_result: EvaluationResult) -> EvaluationResult:
        for index, (simulation_trajectory, experiment_config) in enumerate(
                zip(self._all_measurements, self._experiment_configurations)):
            real_world_trajectory = experiment_config.measurements

            # Align real world timesteps with simulation timesteps
            #   Make simulation timesteps start from 0
            simulation_trajectory[:, 0] -= simulation_trajectory[0, 0]

            # Remove time-step column
            timesteps = simulation_trajectory[:, 0]
            simulation_trajectory = simulation_trajectory[:, 1:]

            # Add acceleration to simulation data
            shoulder_q_vel = simulation_trajectory[:, 2]
            elbow_q_vel = simulation_trajectory[:, 3]

            shoulder_q_acc = np.gradient(shoulder_q_vel, self._environment_config.control_timestep)
            elbow_q_acc = np.gradient(elbow_q_vel, self._environment_config.control_timestep)
            simulation_acceleration = np.vstack([shoulder_q_acc, elbow_q_acc]).T

            # Calculate the load torques
            shoulder_load_torque = calculate_load_torque(joint_index=1,
                                                         torque=simulation_trajectory[:, 4],
                                                         vel=shoulder_q_vel,
                                                         acc=shoulder_q_acc)
            elbow_load_torque = calculate_load_torque(joint_index=2,
                                                      torque=simulation_trajectory[:, 5],
                                                      vel=elbow_q_vel,
                                                      acc=elbow_q_acc)

            simulation_load_torques = np.vstack([shoulder_load_torque, elbow_load_torque]).T

            simulation_trajectory = np.concatenate([simulation_trajectory,
                                                    simulation_acceleration,
                                                    simulation_load_torques], axis=1)

            #  Align the real world measurements with the simulation timesteps
            real_world_trajectory = align_measurements(values=real_world_trajectory,
                                                       simulation_timesteps=timesteps,
                                                       real_world_timesteps=experiment_config.timestep)

            for index, key in enumerate(experiment_config.measurement_keys):
                sim_values = simulation_trajectory[:, index]
                real_values = real_world_trajectory[:, index]

                plt.plot(timesteps, sim_values, color=rgba_green)
                plt.plot(timesteps, real_values, color=rgba_red)
                plt.xlabel('time (s)')
                plt.ylabel(key)
                path = self._base_path / f"genome_{self._genome_id}_episode_" \
                                         f"{experiment_config.identifier}_plot_{key}.png"
                plt.savefig(str(path))
                plt.close()

        return evaluation_result
