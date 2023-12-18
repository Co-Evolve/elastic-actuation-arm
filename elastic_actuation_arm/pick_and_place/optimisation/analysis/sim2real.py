from pathlib import Path
from typing import List

# import matplotlib as mpl
# mpl.use('MacOSX')
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.signal import savgol_filter

from elastic_actuation_arm.calibration.environment.experiment_configuration_handler import align_timesteps1d, to_radians
from elastic_actuation_arm.pick_and_place.optimisation.analysis.real_world_analysis import add_load_torque
from erpy.utils import colors

RW_CONFIGURATION_TO_TIME = {
        "NEA": {
                "go": 1.2320, "return": 2.2960}, "PEA": {
                "go": 3.4400, "return": 4.1320}, "BA": {
                "go": 4.1480, "return": 3.7780}, "FULL": {
                "go": 3.5000, "return": 4.1740}}

SAVGOL_WINDOW_LENGTH = int(0.1 // 0.006)


def smoothen(
        data: np.ndarray,
        window_length: int = SAVGOL_WINDOW_LENGTH
        ) -> np.ndarray:
    return savgol_filter(data, window_length=window_length, polyorder=3)


def load_rw_dataframes(
        base_path: str,
        configuration: str,
        joint_index: int
        ) -> List[pandas.DataFrame]:
    DF_COLUMNS = ["time", "q", "q_vel", "q_acc", "useless1", "torque", "useless2"]

    base_path = Path(base_path)

    phases = ["5kg_go", "0kg_return"]
    dataframes = []
    for phase in phases:
        data_path = base_path / configuration / f"data_file_{joint_index + 1}_{configuration}_{phase}.txt"
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
    DF_COLUMNS = ["time", "q", "q_vel", "q_acc", "useless1", "torque", "useless2", "pea_torque", "ba_torque"]

    base_path = Path(base_path)

    phases = ["go", "return"]
    dataframes = []
    for phase in phases:
        data_path = base_path / configuration / f"{phase}_{joint_name}.csv"
        df = pandas.read_table(
                filepath_or_buffer=data_path, delimiter=' ', names=DF_COLUMNS, skiprows=1
                )

        df["torque"] = smoothen(df["torque"])
        df["q_vel"] = smoothen(df["q_vel"], window_length=int(0.3 // 0.006))
        df["q_acc"] = smoothen(np.gradient(df["q_vel"], df["time"]), window_length=int(0.3 // 0.006))
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
    plt.plot(time, rw_values, color=colors.rgba_red, label=f"rw_{column}")
    plt.plot(time, sm_values, color=colors.rgba_green, label=f"sim_{column}")
    plt.ylabel(column)
    plt.xlabel("time")
    plt.title(f"{configuration}: {joint_name}")


def add_sim_to_current_plot(
        time: np.ndarray,
        sm_values: np.ndarray,
        color: np.ndarray,
        label: str
        ) -> None:
    plt.plot(time, sm_values, color=color, label=label)


def save_current_plot(
        configuration: str,
        column: str,
        joint_name: str,
        legend: bool
        ) -> None:
    if legend:
        plt.legend()
    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["right"].set_visible(False)
    # plt.gca().spines["left"].set_visible(False)
    # plt.gca().spines["bottom"].set_visible(False)
    # plt.gca().yaxis.set_label_position("right")
    go_time = RW_CONFIGURATION_TO_TIME[configuration]["go"]
    return_time = RW_CONFIGURATION_TO_TIME[configuration]["return"]
    # plt.xticks(ticks=[0, go_time, go_time + return_time])
    # max_y = max([max(line.get_ydata()) for line in plt.gca().lines])
    # min_y = min([min(line.get_ydata()) for line in plt.gca().lines])
    # plt.yticks(ticks=[min_y, -0, max_y])
    # if column == "load_torque" and joint_name == "total":
    #     print(f"{configuration} -> {column} -> {joint_name} -> {[min_y, -0, max_y]}")
    # plt.xlim([0, go_time + return_time])
    # plt.ylim([-25, 60])
    plt.savefig(f"./pick_and_place/optimisation/data/sim2real/{configuration}_{column}_{joint_name}.svg")
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
    rw_base_path = "./pick_and_place/optimisation/data/rw_validation"
    sim_base_path = "./pick_and_place/optimisation/data/rw_approximations/best"

    configurations = ["NEA", "PEA", "BA", "FULL"]
    joints = ["shoulder", "elbow"]
    target_plot_columns = ["q", "q_vel", "q_acc", "torque", "load_torque"]
    all_nrmses = {}

    all_torques = {}
    for configuration in configurations:
        for joint_index, joint in enumerate(joints):
            rw_dfs = load_rw_dataframes(
                    base_path=rw_base_path, configuration=configuration, joint_index=joint_index + 1
                    )
            sm_dfs = load_sim_dataframes(
                    base_path=sim_base_path, configuration=configuration, joint_name=joint, joint_index=joint_index + 1
                    )

            # Throw away first 0.1 and last 0.1 seconds
            sm_dfs = [df[(0.1 < df["time"]) & (df["time"] < RW_CONFIGURATION_TO_TIME[configuration][phase] - 0.1)] for
                      phase, df in zip(["go", "return"], sm_dfs)]

            for column in target_plot_columns:
                for i, (rw_df, sm_df) in enumerate(zip(rw_dfs, sm_dfs)):
                    rw_values = rw_df[column].to_numpy()
                    sm_values = sm_df[column].to_numpy()
                    time = sm_df["time"].to_numpy()

                    # align simulation and rw timesteps
                    rw_values = align_timesteps1d(
                            values=rw_values, simulation_timesteps=time, real_world_timesteps=rw_df["time"]
                            )

                    if "load_torque" in column:
                        all_torques[f"{configuration}_phase_{i}_joint_{joint}_rw_load_torque"] = rw_values
                        all_torques[f"{configuration}_phase_{i}_joint_{joint}_sm_load_torque"] = sm_values
                        all_torques[f"{configuration}_phase_{i}_time"] = time + i * \
                                                                         RW_CONFIGURATION_TO_TIME[configuration]["go"]
                        all_torques[f"{configuration}_phase_{i}_joint_{joint}_pea_torque"] = sm_df[
                            "pea_torque"].to_numpy()
                        all_torques[f"{configuration}_phase_{i}_joint_{joint}_ba_torque"] = sm_df[
                            "ba_torque"].to_numpy()

                    error = calculate_nrmse(
                            true=rw_values, pred=sm_values
                            )
                    add_to_current_plot(
                            time=time + i * RW_CONFIGURATION_TO_TIME[configuration]["go"],
                            rw_values=rw_values,
                            sm_values=sm_values,
                            column=column,
                            configuration=configuration,
                            joint_name=joint
                            )
                    if "load_torque" in column:
                        if configuration == "PEA" or configuration == "FULL":
                            add_sim_to_current_plot(
                                    time=time + i * RW_CONFIGURATION_TO_TIME[configuration]["go"],
                                    sm_values=sm_df["pea_torque"].to_numpy(),
                                    color=colors.rgba_orange,
                                    label="pea torque"
                                    )
                        if configuration == "BA" or configuration == "FULL":
                            add_sim_to_current_plot(
                                    time=time + i * RW_CONFIGURATION_TO_TIME[configuration]["go"],
                                    sm_values=sm_df["ba_torque"].to_numpy(),
                                    color=colors.rgba_blue,
                                    label="ba torque"
                                    )
                    all_nrmses[f"configuration_{configuration}_joint_{joint}_{column}_phase_{i}"] = error

                save_current_plot(
                        configuration=configuration, column=column, joint_name=joint, legend=True
                        )

    for configuration in configurations:
        print(f"Configuration: {configuration}")
        for metric in ["q", "q_vel", "q_acc", "torque", "load_torque"]:
            nrmses_per_joint = []
            for joint in joints:
                nrmse = np.average(
                        [all_nrmses[f"configuration_{configuration}_joint_{joint}_{metric}_phase_{phase}"] for phase in
                         [0, 1]]
                        )
                nrmses_per_joint.append(nrmse)

            print(f"\t{metric}:")
            print(f"\t\tShoulder: {round(nrmses_per_joint[0], 4)}")
            print(f"\t\tElbow:    {round(nrmses_per_joint[1], 4)}")
            print(f"\t\tAverage:  {round(np.average(nrmses_per_joint), 4)}")

        # plot total load torques
        for phase in range(2):
            total_rw_lt = np.sum(
                    [all_torques[f"{configuration}_phase_{phase}_joint_{joint}_rw_load_torque"] for joint in joints],
                    axis=0
                    )
            total_sm_lt = np.sum(
                    [all_torques[f"{configuration}_phase_{phase}_joint_{joint}_sm_load_torque"] for joint in joints],
                    axis=0
                    )

            add_to_current_plot(
                    time=all_torques[f"{configuration}_phase_{phase}_time"],
                    rw_values=total_rw_lt,
                    sm_values=total_sm_lt,
                    column="load_torque",
                    configuration=configuration,
                    joint_name="total"
                    )
            if configuration in ["PEA", "FULL"]:
                total_sm_pea = np.sum(
                        [all_torques[f"{configuration}_phase_{phase}_joint_{joint}_pea_torque"] for joint in joints],
                        axis=0
                        )
                add_sim_to_current_plot(
                        time=all_torques[f"{configuration}_phase_{phase}_time"],
                        sm_values=total_sm_pea,
                        color=colors.rgba_orange,
                        label="total pea torque"
                        )
            if configuration in ["BA", "FULL"]:
                total_sm_ba = np.sum(
                        [all_torques[f"{configuration}_phase_{phase}_joint_{joint}_ba_torque"] for joint in joints],
                        axis=0
                        )
                add_sim_to_current_plot(
                        time=all_torques[f"{configuration}_phase_{phase}_time"],
                        sm_values=total_sm_ba,
                        color=colors.rgba_blue,
                        label="total ba torque"
                        )

        save_current_plot(
                configuration=configuration, column="load_torque", joint_name="total", legend=True
                )
