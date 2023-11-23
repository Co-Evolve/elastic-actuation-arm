from pathlib import Path
from typing import List
# import matplotlib as mpl
# mpl.use('MacOSX')
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd

from elastic_actuation_arm.calibration.environment.experiment_configuration_handler import to_radians
from elastic_actuation_arm.pick_and_place.optimisation.analysis.real_world_analysis import add_load_torque
from erpy.utils import colors





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

        # Filter on time
        df = df[(0.5 < df["time"]) & (df["time"] < 2.5)]

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

        # Filter on time
        df = df[(0.5 < df["time"]) & (df["time"] < 2.5)]

        if 'q_acc' not in df:
            df["q_acc"] = np.gradient(df["q_vel"], df["time"])

        add_load_torque(df, joint_index)

        dataframes.append(df)
    return dataframes


def create_plot(
        rw_dfs: List[pd.DataFrame],
        sm_dfs: List[pd.DataFrame],
        column: str,
        configuration: str,
        joint_name: str
        ) -> None:
    for i, (rw_df, sm_df) in enumerate(zip(rw_dfs, sm_dfs)):
        plt.plot(rw_df["time"] + 3 * i, rw_df[column], color=colors.rgba_red)
        plt.plot(sm_df["time"] + 3 * i, sm_df[column], color=colors.rgba_green)
    plt.ylabel(column)
    plt.xlabel("time")
    plt.title(f"{configuration}: {joint_name}")
    plt.savefig(f"./calibration/pap_validation/data/output/{configuration}_{column}_{joint_name}.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    # Create q, qvel, qacc, torque, load torque comparisons between sim and real
    rw_base_path = "./calibration/pap_validation/data/rw"
    sim_base_path = "./calibration/pap_validation/data/sim"

    configurations = ["NEA", "PEA", "BPEA", "FULL"]
    joints = ["shoulder", "elbow"]
    target_plot_columns = ["q", "q_vel", "q_acc", "torque", "load_torque"]
    for configuration in configurations:
        for joint_index, joint in enumerate(joints):
            rw_dfs = load_rw_dataframes(
                    base_path=rw_base_path, configuration=configuration, joint_index=joint_index + 1
                    )
            sm_dfs = load_sim_dataframes(
                base_path=sim_base_path, configuration=configuration, joint_name=joint, joint_index=joint_index + 1
                )

            for column in target_plot_columns:
                create_plot(rw_dfs=rw_dfs, sm_dfs=sm_dfs, column=column, configuration=configuration, joint_name=joint)
