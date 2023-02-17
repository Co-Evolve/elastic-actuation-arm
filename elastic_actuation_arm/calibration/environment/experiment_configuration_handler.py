from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas
from scipy.interpolate import interp1d

from elastic_actuation_arm.entities.manipulator.elastic_actuation import calculate_load_torque


@dataclass
class CalibrationExperimentConfig:
    identifier: int
    high_to_low: bool
    payload: bool
    spring_stiffness_factor: float

    real_world_start_timestep: float
    timestep: np.ndarray
    q: np.ndarray
    q_vel: np.ndarray
    q_acc: np.ndarray
    torque: np.ndarray
    _load_torque: np.ndarray = None

    @property
    def q0(self) -> np.ndarray:
        return self.q[0]

    @property
    def q_vel0(self) -> np.ndarray:
        return self.q_vel[0]

    @property
    def q_acc0(self) -> np.ndarray:
        return self.q_acc[0]

    @property
    def torque0(self) -> np.ndarray:
        return self.torque[0]

    @property
    def load_torque(self) -> np.ndarray:
        if self._load_torque is None:
            shoulder_load_torques = calculate_load_torque(joint_index=1,
                                                          torque=self.torque[:, 0],
                                                          vel=self.q_vel[:, 0],
                                                          acc=self.q_acc[:, 0])
            elbow_load_torques = calculate_load_torque(joint_index=2,
                                                       torque=self.torque[:, 1],
                                                       vel=self.q_vel[:, 1],
                                                       acc=self.q_acc[:, 1])
            self._load_torque = np.vstack((shoulder_load_torques, elbow_load_torques)).T

        return self._load_torque

    @property
    def measurement_keys(self) -> List[str]:
        return ["shoulder_q", "elbow_q",
                "shoulder_q_vel", "elbow_q_vel",
                "shoulder_torque", "elbow_torque",
                "shoulder_q_acc", "elbow_q_acc",
                "shoulder_load_torque", "elbow_load_torque"]

    @property
    def measurements(self) -> np.ndarray:
        return np.concatenate([self.q, self.q_vel, self.torque, self.q_acc, self.load_torque], axis=1)


def concatenate_columns(df1: pandas.DataFrame, df2: pandas.DataFrame, column: str) -> np.ndarray:
    return np.vstack([df1[column].to_numpy(),
                      df2[column].to_numpy()]).T


def to_radians(degrees: np.ndarray) -> np.ndarray:
    return degrees / 180 * np.pi


def align_timesteps2d(values: np.ndarray, simulation_timesteps: np.ndarray, real_world_timesteps: np.ndarray):
    new_values = []
    for i in range(2):
        f = interp1d(x=real_world_timesteps[:, i], y=values[:, i], kind='cubic')
        new_values.append(f(simulation_timesteps))

    return np.array(new_values).T


def align_timesteps1d(values: np.ndarray, simulation_timesteps: np.ndarray, real_world_timesteps: np.ndarray):
    f = interp1d(x=real_world_timesteps, y=values, kind='cubic')
    return np.array(f(simulation_timesteps))


def align_measurements(values: np.ndarray, simulation_timesteps: np.ndarray,
                       real_world_timesteps: np.ndarray) -> np.ndarray:
    aligned_values = np.zeros((simulation_timesteps.shape[0], values.shape[1]))
    for i in range(values.shape[1]):
        rw_timesteps = real_world_timesteps[:, i % 2]  # alternate between shoulder and elbow timesteps
        aligned_values[:, i] = align_timesteps1d(values=values[:, i],
                                                 simulation_timesteps=simulation_timesteps,
                                                 real_world_timesteps=rw_timesteps)
    return aligned_values


def create_calibration_experiment_configs(base_path: str,
                                          experiment_ids: List[int]) -> List[CalibrationExperimentConfig]:
    base_path = Path(base_path)
    COLUMNS = ["time", "q", "q_vel", "q_acc", "useless1", "torque", "useless2"]

    high_to_low = [True, True, True, True, True, True, False, False, False, False, False, False]
    payload = [True, True, True, False, False, False, True, True, True, False, False, False]
    spring_stiffness_factors = [1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0, 1, 0.5, 0]
    identifiers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    experiment_configs = []
    for i, hl, pl, ssf in zip(identifiers, high_to_low, payload, spring_stiffness_factors):
        if i not in experiment_ids:
            continue

        shoulder_data = pandas.read_table(filepath_or_buffer=str(base_path / f"data_file_2_Exp{i}.txt"),
                                          delimiter=' ', names=COLUMNS)
        elbow_data = pandas.read_table(filepath_or_buffer=str(base_path / f"data_file_3_Exp{i}.txt"),
                                       delimiter=' ', names=COLUMNS)

        timesteps = concatenate_columns(shoulder_data, elbow_data, 'time')

        # Shift all time-steps such that real-world data starts from the first timestep of the most delayed joint
        real_world_start_timestep = np.max(timesteps[0, :])
        timesteps[:] -= real_world_start_timestep

        experiment_config = CalibrationExperimentConfig(identifier=i,
                                                        high_to_low=hl,
                                                        payload=pl,
                                                        spring_stiffness_factor=ssf,
                                                        timestep=timesteps,
                                                        real_world_start_timestep=real_world_start_timestep,
                                                        q=to_radians(
                                                            concatenate_columns(shoulder_data, elbow_data, 'q')),
                                                        q_vel=to_radians(
                                                            concatenate_columns(shoulder_data, elbow_data, 'q_vel')),
                                                        q_acc=to_radians(
                                                            concatenate_columns(shoulder_data, elbow_data, 'q_acc')),
                                                        torque=concatenate_columns(shoulder_data, elbow_data, 'torque'))

        experiment_configs.append(experiment_config)

    return experiment_configs


def extract_closest_sim2real_samples(simulation_measurements: List[np.ndarray],
                                     experiment_configurations: List[CalibrationExperimentConfig]) -> List[np.ndarray]:
    """
    Extracts the measurements from the real-world data for which the time-steps are closest to those of the simulated
    data.
    Expects the first column of a measurements array to be the time-step column.
    :param simulation_measurements:
    :param experiment_configurations:
    :return:
    """
    real_world_trajectories = []
    for simulation_trajectory, experiment_config in zip(simulation_measurements, experiment_configurations):
        real_world_measurements = experiment_config.measurements
        real_world_trajectory = []
        real_world_index = 0

        for sample in simulation_trajectory:
            sim_timestep = sample[0]

            while real_world_index + 1 < len(experiment_config.timestep):
                # Do a sum here -- real world data has different time steps per joint -> summing distance
                #   results in choosing the real world measurement with time steps as closest as possible for both
                current_distance = sum(abs(sim_timestep - experiment_config.timestep[real_world_index]))
                next_distance = sum(abs(sim_timestep - experiment_config.timestep[real_world_index + 1]))

                if next_distance > current_distance:
                    break
                else:
                    real_world_index += 1

            real_world_trajectory.append(real_world_measurements[real_world_index])

        real_world_trajectory = np.array(real_world_trajectory)
        real_world_trajectories.append(real_world_trajectory)

    return real_world_trajectories
