from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy
from elastic_actuation_arm.pick_and_place.optimization.spring.analysis.orig_opt_comparison import \
    add_load_torque

from elastic_actuation_arm.calibration.environment.experiment_configuration_handler import to_radians
from erpy.utils import colors

DF_COLUMNS = ["time", "q", "q_vel", "q_acc", "useless1", "torque", "useless2"]

CONFIGURATION_TO_TIME = {
    "NEA": {
        "go": 1.2320,
        "return": 2.2360
    },
    "PEA": {
        "go": 3.4400,
        "return": 4.1320
    },
    "BA": {
        "go": 4.5300,
        "return": 3.3780
    },
    "PEA_and_BA": {
        "go": 4.0280,
        "return": 4.4320
    }
}


def load_rw_dataframes(path: str, configuration: str) -> List[pandas.DataFrame]:
    base_path = Path(path)

    phases = ["5kg_go", "0kg_return"]
    dataframes = []
    for joint_index in [1, 2]:
        dataframes_per_joint = []
        time_shift = 0
        for phase in phases:
            data_path = base_path / configuration / f"data_file_{joint_index + 1}_{configuration}_{phase}.txt"
            df = pandas.read_table(filepath_or_buffer=data_path,
                                   delimiter=' ', names=DF_COLUMNS)

            for column in ['q', 'q_vel', 'q_acc']:
                df[column] = to_radians(df[column].to_numpy())

            # Filter on time
            ph = "go" if "go" in phase else "return"
            max_time = CONFIGURATION_TO_TIME[configuration][ph]
            cutoff = max_time * 0.0
            df = df[(cutoff < df["time"]) & (df["time"] < (max_time - cutoff))]

            # For concatenation
            df["time"] += time_shift
            time_shift = np.max(df["time"])

            add_load_torque(df, joint_index)
            dataframes_per_joint.append(df)

        df = pandas.concat(dataframes_per_joint)
        dataframes.append(df)

    return dataframes


def load_sm_dataframes(path: str, configuration: str) -> List[pandas.DataFrame]:
    base_path = Path(path)

    phases = ["go", "return"]
    dataframes = []
    for joint_name, joint_index in zip(["shoulder", "elbow"], [1, 2]):
        dataframes_per_joint = []
        time_shift = 0
        for phase in phases:
            data_path = base_path / configuration / f"{phase}_{joint_name}.csv"

            df = pandas.read_table(filepath_or_buffer=data_path,
                                   delimiter=' ', names=DF_COLUMNS)

            # Skip first value
            df = df.tail(-1)

            # Filter on time
            max_time = CONFIGURATION_TO_TIME[configuration][phase]
            cutoff = max_time * 0.0
            df = df[(cutoff < df["time"]) & (df["time"] < (max_time - cutoff))]

            df["time"] += time_shift
            time_shift = np.max(df["time"])

            add_load_torque(df, joint_index)
            dataframes_per_joint.append(df)

        # Concatenate dataframes
        df = pandas.concat(dataframes_per_joint)
        dataframes.append(df)

    return dataframes


def calculate_lts_integral(dataframe: pandas.DataFrame) -> float:
    timesteps = dataframe["time"]
    load_torque = dataframe["load_torque"].to_numpy()
    load_torque_squared = np.power(load_torque, 2)
    integral = scipy.integrate.simpson(y=load_torque_squared,
                                       x=timesteps)
    return integral


def print_values(shoulder_lts, elbow_lts, shoulder_df, elbow_df, nea_val):
    print(f"\t\tShoulder:             {shoulder_lts:4f}")
    print(f"\t\tElbow:                {elbow_lts:4f}")
    print(f"\t\tTotal:                {(shoulder_lts + elbow_lts):4f}")
    print(f"\t\tTotal time:           {max(np.max(shoulder_df['time']), np.max(elbow_df['time']))}")
    print(f"\t\tRatio w.r.t. NEA:     {(((shoulder_lts + elbow_lts) / nea_val) * 100):2f}")


def plot_data(title, rw_df, sm_df, key="load_torque"):
    plt.plot(rw_df["time"], rw_df[key], color=colors.rgba_red)
    plt.plot(sm_df["time"], sm_df[key], color=colors.rgba_green)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(key)
    if not Path(f"./sim2real_debug_plots/{key}").exists():
        Path(f"./sim2real_debug_plots/{key}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"./sim2real_debug_plots/{key}/{title}.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    rw_base_path = Path("./output/rw-validation")
    sm_base_path = Path("./output/cma-rw-values-v1/best")
    sm_fc_base_path = Path("./output/cma-final/best")
    configurations = ["NEA", "PEA", "BA", "PEA_and_BA"]

    rw_nea_val = None
    sm_nea_val = None
    for configuration in configurations:
        rw_shoulder_df, rw_elbow_df = load_rw_dataframes(path=rw_base_path, configuration=configuration)
        rw_shoulder_lts = calculate_lts_integral(rw_shoulder_df)
        rw_elbow_lts = calculate_lts_integral(rw_elbow_df)

        sm_shoulder_df, sm_elbow_df = load_sm_dataframes(path=sm_base_path, configuration=configuration)
        sm_shoulder_lts = calculate_lts_integral(sm_shoulder_df)
        sm_elbow_lts = calculate_lts_integral(sm_elbow_df)

        sm_fc_shoulder_df, sm_fc_elbow_df = load_sm_dataframes(path=sm_fc_base_path, configuration=configuration)
        sm_fc_shoulder_lts = calculate_lts_integral(sm_fc_shoulder_df)
        sm_fc_elbow_lts = calculate_lts_integral(sm_fc_elbow_df)

        if configuration == "NEA":
            rw_nea_val = rw_shoulder_lts + rw_elbow_lts
            sm_nea_val = sm_shoulder_lts + sm_elbow_lts

        print(f"Configuration:          {configuration}")
        print(f"\tReal-world")
        print_values(shoulder_lts=rw_shoulder_lts, elbow_lts=rw_elbow_lts, shoulder_df=rw_shoulder_df,
                     elbow_df=rw_elbow_df, nea_val=rw_nea_val)
        print(f"\tSimulation")
        print_values(shoulder_lts=sm_shoulder_lts, elbow_lts=sm_elbow_lts, shoulder_df=sm_shoulder_df,
                     elbow_df=sm_elbow_df, nea_val=sm_nea_val)
        print(f"\tSimulation (FULLY CONTINUOUS)")
        print_values(shoulder_lts=sm_fc_shoulder_lts, elbow_lts=sm_fc_elbow_lts, shoulder_df=sm_fc_shoulder_df,
                     elbow_df=sm_fc_elbow_df, nea_val=sm_nea_val)
        print("-" * 50)

        key = "load_torque"
        plot_data(f"{configuration}_shoulder", rw_shoulder_df, sm_shoulder_df, key=key)
        plot_data(f"{configuration}_elbow", rw_elbow_df, sm_elbow_df, key=key)
