from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from elastic_actuation_arm.pick_and_place.optimisation.analysis.real_world_analysis import load_sm_dataframes, \
    RW_CONFIGURATION_TO_TIME

mpl.use('MacOSX')

from erpy.utils import colors

if __name__ == '__main__':
    path = Path("../output/cma-rw-values/best")
    shoulder_df, elbow_df = load_sm_dataframes(path=path, configuration="PEA_and_BA", fc=False)

    shoulder_pea, elbow_pea = shoulder_df["pea_torque"].to_numpy(), elbow_df["pea_torque"].to_numpy()
    total_pea = shoulder_pea + elbow_pea

    shoulder_bea, elbow_bea = shoulder_df["bea_torque"].to_numpy(), elbow_df["bea_torque"].to_numpy()
    total_bea = shoulder_bea + elbow_bea

    shoulder_lt, elbow_lt = shoulder_df["load_torque"].to_numpy(), elbow_df["load_torque"].to_numpy()
    total_lt = shoulder_lt + elbow_lt

    shoulder_time = shoulder_df["time"].to_numpy()
    elbow_time = elbow_df["time"].to_numpy()
    plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
    plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
    plt.rcParams["figure.figsize"] = (4, 3)

    plt.plot(shoulder_time, total_lt, color=colors.rgba_gray)
    plt.plot(shoulder_time, total_pea, color=colors.rgba_green)
    plt.plot(shoulder_time, total_bea, color=colors.rgba_red)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().yaxis.set_label_position("right")
    go_time = RW_CONFIGURATION_TO_TIME["PEA_and_BA"]["go"]
    return_time = RW_CONFIGURATION_TO_TIME["PEA_and_BA"]["return"]
    plt.xticks(ticks=[0, go_time, max(shoulder_time)], labels=["", "", ""])
    all_values = np.concatenate((total_lt, total_pea, total_bea))
    plt.yticks(ticks=[min(all_values), 0, max(all_values)], labels=["", "", ""])
    print([min(all_values), 0, max(all_values)])
    plt.xlim([0, max(shoulder_time)])
    plt.savefig("torque_total.svg")
    plt.show()
    plt.close()

    plt.plot(shoulder_time, shoulder_lt, color=colors.rgba_gray)
    plt.plot(shoulder_time, shoulder_pea, color=colors.rgba_green)
    plt.plot(shoulder_time, shoulder_bea, color=colors.rgba_red)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().yaxis.set_label_position("right")
    go_time = RW_CONFIGURATION_TO_TIME["PEA_and_BA"]["go"]
    return_time = RW_CONFIGURATION_TO_TIME["PEA_and_BA"]["return"]
    plt.xticks(ticks=[0, go_time, max(shoulder_time)], labels=["", "", ""])
    all_values = np.concatenate((shoulder_lt, shoulder_pea, shoulder_bea))
    plt.yticks(ticks=[min(all_values), 0, max(all_values)], labels=["", "", ""])
    print([min(all_values), 0, max(all_values)])
    plt.savefig("torque_shoulder.svg")
    plt.show()
    plt.close()

    plt.plot(elbow_time, elbow_lt, color=colors.rgba_gray)
    plt.plot(elbow_time, elbow_pea, color=colors.rgba_green)
    plt.plot(elbow_time, elbow_bea, color=colors.rgba_red)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().yaxis.set_label_position("right")
    go_time = RW_CONFIGURATION_TO_TIME["PEA_and_BA"]["go"]
    return_time = RW_CONFIGURATION_TO_TIME["PEA_and_BA"]["return"]
    plt.xticks(ticks=[0, go_time, max(shoulder_time)], labels=["", "", ""])
    all_values = np.concatenate((elbow_lt, elbow_pea, elbow_bea))
    plt.yticks(ticks=[min(all_values), 0, max(all_values)], labels=["", "", ""])
    print([min(all_values), 0, max(all_values)])
    plt.savefig("torque_elbow.svg")
    plt.show()
    plt.close()
