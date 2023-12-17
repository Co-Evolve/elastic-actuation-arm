from pathlib import Path

# import matplotlib as mpl
# mpl.use('MacOSX')
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.signal import savgol_filter

from elastic_actuation_arm.calibration.environment.experiment_configuration_handler import align_timesteps1d, to_radians
from elastic_actuation_arm.pick_and_place.optimisation.analysis.real_world_analysis import add_load_torque
from erpy.utils import colors

SAVGOL_WINDOW_LENGTH = int(3 // 0.006)


def smoothen(
        data: np.ndarray
        ) -> np.ndarray:
    return savgol_filter(data, window_length=SAVGOL_WINDOW_LENGTH, polyorder=3)


def load_rw_dataframe(
        base_path: str,
        trajectory_id: int,
        joint_index: int
        ) -> pandas.DataFrame:
    DF_COLUMNS = ["time", "q", "q_vel", "q_acc", "useless1", "torque", "useless2"]

    base_path = Path(base_path)

    data_path = base_path / f"data_file_{joint_index + 1}_Exp{trajectory_id}.txt"
    df = pandas.read_table(
            filepath_or_buffer=data_path, delimiter=' ', names=DF_COLUMNS
            )

    for column in ['q', 'q_vel', 'q_acc']:
        df[column] = to_radians(df[column].to_numpy())

    df["torque"] = smoothen(df["torque"])
    df["q_vel"] = smoothen(df["q_vel"])
    df["q_acc"] = smoothen(np.gradient(df["q_vel"], df["time"]))

    add_load_torque(df, joint_index)
    return df


def load_sim_dataframe(
        base_path: str,
        trajectory_id: int,
        joint_index: int
        ) -> pandas.DataFrame:
    DF_COLUMNS = ["time", "q", "q_vel", "torque"]

    base_path = Path(base_path)

    data_path = base_path / f"joint_{joint_index}_trajectory_{trajectory_id}.csv"
    df = pandas.read_table(
            filepath_or_buffer=data_path, delimiter=' ', names=DF_COLUMNS, skiprows=1
            )

    df["torque"] = smoothen(df["torque"])
    df["q_vel"] = smoothen(df["q_vel"])
    df["q_acc"] = smoothen(np.gradient(df["q_vel"], df["time"]))
    add_load_torque(df, joint_index)

    return df


def create_plot(
        time: np.ndarray,
        rw_values: np.ndarray,
        sm_values: np.ndarray,
        column: str,
        trajectory_id: int,
        joint_name: str
        ) -> None:
    plt.plot(
            time, rw_values, color=colors.rgba_red
            )
    plt.plot(
            time, sm_values, color=colors.rgba_green
            )
    plt.ylabel(column)
    plt.xlabel("time")
    plt.title(f"Trajectory {trajectory_id} - {joint_name}")
    plt.savefig(f"./calibration/optimisation/data/sim2real_plots/trajectory_{trajectory_id}_{column}_{joint_name}.png")
    # plt.show()
    plt.close()


def calculate_nrmse(
        true: np.ndarray,
        pred: np.ndarray
        ) -> float:
    mse = np.average(np.power(true - pred, 2))
    nmse = mse / np.var(true)
    return np.sqrt(nmse)


if __name__ == '__main__':
    # Create q, qvel, qacc, torque, load torque comparisons between sim and real
    rw_base_path = "./calibration/environment/real_world_data/version2"
    sim_base_path = "./calibration/optimisation/data/trajectory_data"

    joints = ["shoulder", "elbow"]
    target_plot_columns = ["q", "q_vel", "q_acc", "torque", "load_torque"]
    trajectory_ids = [1, 3, 4, 6, 7, 9, 10, 12]

    all_nrmses = {}
    for trajectory_id in trajectory_ids:
        print(f"Trajectory: {trajectory_id}")
        for joint_index, joint in enumerate(joints):
            print(f"\tJoint: {joint}")
            rw_df = load_rw_dataframe(
                    base_path=rw_base_path, trajectory_id=trajectory_id, joint_index=joint_index + 1
                    )
            sm_df = load_sim_dataframe(
                    base_path=sim_base_path, trajectory_id=trajectory_id, joint_index=joint_index + 1
                    )

            # Throw away first 5 and last 5 seconds
            sm_df = sm_df[(5 < sm_df["time"]) & (sm_df["time"] < 55)]

            for column in target_plot_columns:
                rw_values = rw_df[column].to_numpy()
                sm_values = sm_df[column].to_numpy()
                time = sm_df["time"].to_numpy()

                # align simulation and rw timesteps
                rw_values = align_timesteps1d(
                        values=rw_values, simulation_timesteps=time, real_world_timesteps=rw_df["time"]
                        )

                error = calculate_nrmse(true=rw_values, pred=sm_values)

                # Create plots
                create_plot(
                        time=time,
                        rw_values=rw_values,
                        sm_values=sm_values,
                        column=column,
                        trajectory_id=trajectory_id,
                        joint_name=joint
                        )

                print(f"\t\t{column}:\t\t{error:4f}")
                all_nrmses[f"trajectory_{trajectory_id}_joint_{joint_index}_{column}"] = error  # shifts = list(

    trajectory_ids = [1, 3, 4, 6, 7, 9, 10, 12]
    SA_trajectories = [3, 6, 9, 12]
    PEA_trajectories = [1, 4, 7, 10]
    for metric in ["q", "q_vel", "q_acc", "torque", "load_torque"]:
        SA_shoulder = round(
                np.average(
                        [all_nrmses[f"trajectory_{trajectory_id}_joint_0_{metric}"] for trajectory_id in
                         SA_trajectories]
                        ), 4
                )
        SA_elbow = round(
                np.average(
                        [all_nrmses[f"trajectory_{trajectory_id}_joint_1_{metric}"] for trajectory_id in
                         SA_trajectories]
                        ), 4
                )
        PEA_shoulder = round(
                np.average(
                        [all_nrmses[f"trajectory_{trajectory_id}_joint_0_{metric}"] for trajectory_id in
                         PEA_trajectories]
                        ), 4
                )
        PEA_elbow = round(
                np.average(
                        [all_nrmses[f"trajectory_{trajectory_id}_joint_1_{metric}"] for trajectory_id in
                         PEA_trajectories]
                        ), 4
                )

        print(f"{metric}")
        print(f"\tSA:")
        print(f"\t\tShoulder:  {SA_shoulder}")
        print(f"\t\tElbow:     {SA_elbow}")
        print(f"\t\tAverage:   {round(np.average([SA_shoulder, SA_elbow]), 4)}")
        print(f"\tPEA:")
        print(f"\t\tShoulder:  {PEA_shoulder}")
        print(f"\t\tElbow:     {PEA_elbow}")
        print(f"\t\tAverage:   {round(np.average([PEA_shoulder, PEA_elbow]), 4)}")
