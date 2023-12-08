from pathlib import Path
from typing import List, Tuple

# import matplotlib as mpl
# mpl.use('MacOSX')
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.signal import savgol_filter

from elastic_actuation_arm.calibration.environment.experiment_configuration_handler import align_timesteps1d, to_radians
from elastic_actuation_arm.pick_and_place.optimisation.analysis.real_world_analysis import add_load_torque
from erpy.utils import colors

SAVGOL_WINDOW_LENGTH = int(0.25 // 0.006)


def smoothen(
        data: np.ndarray
        ) -> np.ndarray:
    return savgol_filter(data, window_length=SAVGOL_WINDOW_LENGTH, polyorder=3)


def load_rw_dataframes(
        base_path: str,
        configuration: str,
        joint_index: int
        ) -> List[pandas.DataFrame]:
    DF_COLUMNS = ["time", "q", "q_vel", "q_acc", "useless1", "torque", "useless2"]

    base_path = Path(base_path)

    phases = ["5kg_3s_go", "0kg_3s_return"]
    dataframes = []
    for phase in phases:
        data_path = base_path / configuration / f"data_file_{joint_index + 1}_{phase}.txt"
        df = pandas.read_table(
                filepath_or_buffer=data_path, delimiter=' ', names=DF_COLUMNS
                )

        for column in ['q', 'q_vel', 'q_acc']:
            df[column] = to_radians(df[column].to_numpy())

        df["torque"] = smoothen(df["torque"])
        df["q_vel"] = smoothen(df["q_vel"])
        df["q_acc"] = smoothen(np.gradient(df["q_vel"], df["time"]))

        add_load_torque(df, joint_index)

        dataframes.append(df)
    return dataframes


def load_sim_dataframes(
        base_path: str,
        configuration: str,
        joint_name: str,
        joint_index: int, ) -> List[pandas.DataFrame]:
    DF_COLUMNS = ["time", "q", "q_vel", "torque"]

    base_path = Path(base_path)

    phases = ["go", "return"]
    dataframes = []
    for phase in phases:
        data_path = base_path / configuration / f"{joint_name}_{phase}.csv"
        df = pandas.read_table(
                filepath_or_buffer=data_path, delimiter=' ', names=DF_COLUMNS, skiprows=1
                )

        df["torque"] = smoothen(df["torque"])
        df["q_vel"] = smoothen(df["q_vel"])
        df["q_acc"] = smoothen(np.gradient(df["q_vel"], df["time"]))

        add_load_torque(df, joint_index)

        dataframes.append(df)
    return dataframes


def add_to_current_plot(
        time: np.ndarray,
        rw_values: np.ndarray,
        sm_values: np.ndarray,
        column: str,
        configuration: str,
        joint_name: str
        ) -> None:
    plt.plot(time, rw_values, color=colors.rgba_red)
    plt.plot(time, sm_values, color=colors.rgba_green)
    plt.ylabel(column)
    plt.xlabel("time")
    plt.title(f"{configuration}: {joint_name}")


def save_current_plot(
        configuration: str,
        column: str,
        joint_name: str
        ) -> None:
    plt.savefig(f"./calibration/pap_validation/data/sim2real/{configuration}_{column}_{joint_name}.png")
    plt.close()


def calculate_nrmse(
        true: np.ndarray,
        pred: np.ndarray
        ) -> float:
    mse = np.average(np.power(true - pred, 2))
    nmse = mse / np.var(true)
    return np.sqrt(nmse)


def window_shift(
        time: np.ndarray,
        sm_values: np.ndarray,
        rw_values: np.ndarray,
        shift: int
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shift < 0:
        shifted_time = time[abs(shift):]
        shifted_sm_values = sm_values[abs(shift):]
        shifted_rw_values = rw_values[:shift]
    elif shift > 0:
        shifted_time = time[:-shift]
        shifted_sm_values = sm_values[:-shift]
        shifted_rw_values = rw_values[shift:]
    else:
        shifted_time = time
        shifted_sm_values = sm_values
        shifted_rw_values = rw_values
    return shifted_time, shifted_sm_values, shifted_rw_values


if __name__ == '__main__':
    # Create q, qvel, qacc, torque, load torque comparisons between sim and real
    rw_base_path = "./calibration/pap_validation/data/rw"
    sim_base_path = "./calibration/pap_validation/data/sim"

    configurations = ["NEA", "PEA", "BPEA", "FULL"]
    joints = ["shoulder", "elbow"]
    target_plot_columns = ["q", "q_vel", "q_acc", "torque", "load_torque"]
    all_nrmses = {}
    for configuration in configurations:
        for joint_index, joint in enumerate(joints):
            rw_dfs = load_rw_dataframes(
                    base_path=rw_base_path, configuration=configuration, joint_index=joint_index + 1
                    )
            sm_dfs = load_sim_dataframes(
                    base_path=sim_base_path, configuration=configuration, joint_name=joint, joint_index=joint_index + 1
                    )

            # Throw away first 0.5 and last 0.5 seconds
            sm_dfs = [sm_df[(0.5 < sm_df["time"]) & (sm_df["time"] < 2.5)] for sm_df in sm_dfs]

            for column in target_plot_columns:
                for i, (rw_df, sm_df) in enumerate(zip(rw_dfs, sm_dfs)):
                    rw_values = rw_df[column].to_numpy()
                    sm_values = sm_df[column].to_numpy()
                    time = sm_df["time"].to_numpy()

                    # align simulation and rw timesteps
                    rw_values = align_timesteps1d(
                            values=rw_values, simulation_timesteps=time, real_world_timesteps=rw_df["time"]
                            )

                    # Shift
                    window_size_in_seconds = 0.5
                    dt = 0.006
                    window_size = int(window_size_in_seconds // dt)
                    shifts = list(range(-window_size, window_size))
                    minimum_error = np.inf
                    best_shift = None
                    for shift in shifts:
                        _, shifted_sm_values, shifted_rw_values = window_shift(
                                time=time, sm_values=sm_values, rw_values=rw_values, shift=shift
                                )

                        # Calculate NRMSE
                        nrmse = calculate_nrmse(
                                true=shifted_rw_values, pred=shifted_sm_values
                                )
                        if nrmse < minimum_error:
                            best_shift = shift
                            minimum_error = nrmse

                    time, sm_values, rw_values = window_shift(
                            time=time, sm_values=sm_values, rw_values=rw_values, shift=best_shift
                            )
                    add_to_current_plot(time=time + i * 3,
                                        rw_values=rw_values,
                                        sm_values=sm_values,
                                        column=column,
                                        configuration=configuration,
                                        joint_name=joint)
                    all_nrmses[f"configuration_{configuration}_joint_{joint}_{column}_phase_{i}"] = minimum_error

                save_current_plot(configuration=configuration,
                                  column=column,
                                  joint_name=joint)

    for configuration in configurations:
        print(f"Configuration: {configuration}")
        for joint in joints:
            nrmse = np.average([all_nrmses[f"configuration_{configuration}_joint_{joint}_load_torque_phase_{phase}"]
            for phase in [0,1]])
            print(f"\t{joint}:\t\t{nrmse:4f}")
