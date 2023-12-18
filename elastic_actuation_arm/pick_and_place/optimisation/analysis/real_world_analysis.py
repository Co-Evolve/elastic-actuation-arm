from pathlib import Path
from typing import List

import numpy as np
import pandas
import scipy

from elastic_actuation_arm.calibration.environment.experiment_configuration_handler import to_radians
from elastic_actuation_arm.entities.manipulator.elastic_actuation import calculate_load_torque
from erpy.utils import colors

DF_COLUMNS = ["time", "q", "q_vel", "q_acc", "useless1", "torque", "useless2"]

RW_CONFIGURATION_TO_TIME = {
    "NEA": {
        "go": 1.2320,
        "return": 2.2960
    },
    "PEA": {
        "go": 3.4400,
        "return": 4.1320
    },
    "BA": {
        "go": 4.1480,
        "return": 3.7780
    },
    "PEA_and_BA": {
        "go": 3.5000,
        "return": 4.1740
    }
}

FC_CONFIGURATION_TO_TIME = {
    "NEA": {
        "go": 1.2320,
        "return": 2.2960
    },
    "PEA": {
        "go": 3.3980,
        "return": 4.1260
    },
    "BA": {
        "go": 4.4897,
        "return": 3.5141
    },
    "PEA_and_BA": {
        "go": 3.7039,
        "return": 4.7141
    }
}


def add_load_torque(dataframe: pandas.DataFrame, joint_index: int, column_name: str = "load_torque") -> None:
    load_torques = calculate_load_torque(joint_index=joint_index,
                                         torque=dataframe['torque'].to_numpy(),
                                         vel=dataframe['q_vel'].to_numpy(),
                                         acc=dataframe['q_acc'].to_numpy()
                                         )

    dataframe[column_name] = load_torques


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
            max_time = RW_CONFIGURATION_TO_TIME[configuration][ph]
            df = df[df["time"] < max_time]

            # For concatenation
            df["time"] += time_shift
            time_shift = np.max(df["time"])

            add_load_torque(df, joint_index)
            dataframes_per_joint.append(df)

        df = pandas.concat(dataframes_per_joint)
        dataframes.append(df)

    return dataframes


def load_sm_dataframes(path: str, configuration: str, fc: bool, separate_dfs: bool = False) -> List[pandas.DataFrame]:
    base_path = Path(path)

    phases = ["go", "return"]
    dataframes = []
    for joint_name, joint_index in zip(["shoulder", "elbow"], [1, 2]):
        dataframes_per_joint = []
        time_shift = 0
        for phase in phases:
            data_path = base_path / configuration / f"{phase}_{joint_name}.csv"

            df = pandas.read_table(filepath_or_buffer=data_path,
                                   delimiter=' ', names=DF_COLUMNS + ["pea_torque", "bea_torque"])

            # Skip first value
            # df = df.tail(-1)

            # Filter on time
            if fc:
                max_time = FC_CONFIGURATION_TO_TIME[configuration][phase]
            else:
                max_time = RW_CONFIGURATION_TO_TIME[configuration][phase]
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


def calculate_lt_rms(df) -> float:
    load_torque = df["load_torque"].to_numpy()
    rms = np.sqrt(np.mean(np.power(load_torque, 2)))
    return rms


def calculate_lt_peak(df) -> float:
    load_torque = np.max(np.abs(df["load_torque"].to_numpy()))
    return load_torque

def print_values(name, value, baseline_value):
    absolute_decrease = baseline_value - value
    relative_decrease = ((absolute_decrease / baseline_value) * 100)
    print(f"\t\t{name}:\t{absolute_decrease:.1f} ({relative_decrease:.0f}%)")

if __name__ == '__main__':
    rw_base_path = Path("./output/rw-validation")
    sm_base_path = Path("./output/cma-rw-values/best")
    sm_fc_base_path = Path("./output/cma-final/best")
    configurations = ["NEA", "PEA", "BA", "PEA_and_BA"]

    rw_nea_shoulder_cost = None
    rw_nea_shoulder_rms = None
    rw_nea_shoulder_peak = None
    rw_nea_elbow_cost = None
    rw_nea_elbow_rms = None
    rw_nea_elbow_peak = None

    sm_nea_shoulder_cost = None
    sm_nea_shoulder_rms = None
    sm_nea_shoulder_peak = None
    sm_nea_elbow_cost = None
    sm_nea_elbow_rms = None
    sm_nea_elbow_peak = None

    for configuration in configurations:
        # RW dataframe
        rw_shoulder_df, rw_elbow_df = load_rw_dataframes(path=rw_base_path, configuration=configuration)

        rw_shoulder_cost = calculate_lts_integral(rw_shoulder_df)
        rw_shoulder_rms = calculate_lt_rms(rw_shoulder_df)
        rw_shoulder_peak = calculate_lt_peak(rw_shoulder_df)

        rw_elbow_cost = calculate_lts_integral(rw_elbow_df)
        rw_elbow_rms = calculate_lt_rms(rw_elbow_df)
        rw_elbow_peak = calculate_lt_peak(rw_elbow_df)

        # * dataframe
        sm_shoulder_df, sm_elbow_df = load_sm_dataframes(path=sm_base_path, configuration=configuration,
                                                         fc=False)
        sm_shoulder_cost = calculate_lts_integral(sm_shoulder_df)
        sm_shoulder_rms = calculate_lt_rms(sm_shoulder_df)
        sm_shoulder_peak = calculate_lt_peak(sm_shoulder_df)

        sm_elbow_cost = calculate_lts_integral(sm_elbow_df)
        sm_elbow_rms = calculate_lt_rms(sm_elbow_df)
        sm_elbow_peak = calculate_lt_peak(sm_elbow_df)

        # Complete optimization
        sm_fc_shoulder_df, sm_fc_elbow_df = load_sm_dataframes(path=sm_fc_base_path, configuration=configuration,
                                                               fc=True)
        sm_fc_shoulder_cost = calculate_lts_integral(sm_fc_shoulder_df)
        sm_fc_shoulder_rms = calculate_lt_rms(sm_fc_shoulder_df)
        sm_fc_shoulder_peak = calculate_lt_peak(sm_fc_shoulder_df)

        sm_fc_elbow_cost = calculate_lts_integral(sm_fc_elbow_df)
        sm_fc_elbow_rms = calculate_lt_rms(sm_fc_elbow_df)
        sm_fc_elbow_peak = calculate_lt_peak(sm_fc_elbow_df)

        if configuration == "NEA":
            rw_nea_shoulder_cost = rw_shoulder_cost
            rw_nea_shoulder_rms = rw_shoulder_rms
            rw_nea_shoulder_peak = rw_shoulder_peak
            rw_nea_elbow_cost = rw_elbow_cost
            rw_nea_elbow_rms = rw_elbow_rms
            rw_nea_elbow_peak = rw_elbow_peak

            sm_nea_shoulder_cost = sm_shoulder_cost
            sm_nea_shoulder_rms = sm_shoulder_rms
            sm_nea_shoulder_peak = sm_shoulder_peak
            sm_nea_elbow_cost = sm_elbow_cost
            sm_nea_elbow_rms = sm_elbow_rms
            sm_nea_elbow_peak = sm_elbow_peak
        else:
            print(f"Configuration:          {configuration}")
            print(f"\tSimulation (FULLY CONTINUOUS)")
            print_values("cost - shoulder", sm_fc_shoulder_cost, sm_nea_shoulder_cost)
            print_values("rms  - shoulder", sm_fc_shoulder_rms, sm_nea_shoulder_rms)
            print_values("peak - shoulder", sm_fc_shoulder_peak, sm_nea_shoulder_peak)
            print()
            print_values("cost - elbow   ", sm_fc_elbow_cost, sm_nea_elbow_cost)
            print_values("rms  - elbow   ", sm_fc_elbow_rms, sm_nea_elbow_rms)
            print_values("peak - elbow   ", sm_fc_elbow_peak, sm_nea_elbow_peak)
            print(f"\tSimulation (*)")
            print_values("cost - shoulder", sm_shoulder_cost, sm_nea_shoulder_cost)
            print_values("rms  - shoulder", sm_shoulder_rms, sm_nea_shoulder_rms)
            print_values("peak - shoulder", sm_shoulder_peak, sm_nea_shoulder_peak)
            print()
            print_values("cost - elbow   ", sm_elbow_cost, sm_nea_elbow_cost)
            print_values("rms  - elbow   ", sm_elbow_rms, sm_nea_elbow_rms)
            print_values("peak - elbow   ", sm_elbow_peak, sm_nea_elbow_peak)
            print(f"\tReal-world")
            print_values("cost - shoulder", rw_shoulder_cost, rw_nea_shoulder_cost)
            print_values("rms  - shoulder", rw_shoulder_rms, rw_nea_shoulder_rms)
            print_values("peak - shoulder", rw_shoulder_peak, rw_nea_shoulder_peak)
            print()
            print_values("cost - elbow   ", rw_elbow_cost, rw_nea_elbow_cost)
            print_values("rms  - elbow   ", rw_elbow_rms, rw_nea_elbow_rms)
            print_values("peak - elbow   ", rw_elbow_peak, rw_nea_elbow_peak)
            print("-" * 50)


        if configuration == "PEA":
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            mpl.use('MacOSX')

            plt.plot(rw_shoulder_df["time"], rw_shoulder_df["load_torque"], color=colors.rgba_green)
            plt.plot(sm_shoulder_df["time"], sm_shoulder_df["load_torque"], color=colors.rgba_red)
            plt.show()
            plt.close()
            plt.plot(rw_shoulder_df["time"], np.power(rw_shoulder_df["load_torque"], 2), color=colors.rgba_green)
            plt.plot(sm_shoulder_df["time"], np.power(sm_shoulder_df["load_torque"], 2), color=colors.rgba_red)
            plt.show()