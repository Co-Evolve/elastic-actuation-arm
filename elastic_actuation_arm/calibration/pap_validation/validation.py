from pathlib import Path

import numpy as np
import pandas as pd
from dm_control import viewer
from dm_env import TimeStep

from elastic_actuation_arm.calibration.pap_validation.robot.robot import ManipulatorCalibrationValidationRobot
from elastic_actuation_arm.calibration.pap_validation.robot.specifications import (bea_specification,
                                                                                   full_specification, \
                                                                                   nea_specification, pea_specification)
from elastic_actuation_arm.pick_and_place.environment.environment import PickAndPlaceEnvironmentConfig, \
    parse_pickandplace_environment_observations
from erpy.base.specification import RobotSpecification
from erpy.interfaces.mujoco.gym_wrapper import _flatten_obs


def get_environment_config() -> PickAndPlaceEnvironmentConfig:
    return PickAndPlaceEnvironmentConfig(
            seed=42,
            random_state=np.random.RandomState(seed=42),
            contacts_enabled=True,
            initialisation_duration=4,
            go_duration=3,
            placement_duration=2,
            ret_duration=3
            )


def record_episode_statistics(
        specification: RobotSpecification,
        output_path: str
        ) -> pd.DataFrame:
    robot = ManipulatorCalibrationValidationRobot(specification=specification)
    env = get_environment_config().environment(robot=robot, wrap2gym=True)

    # shoulder df -> go phase, return phasae
    # elbow df -> go phase, return phase
    all_obs = []
    done = False
    obs = env.reset()
    while not done:
        all_obs.append(obs)
        actions = robot(obs)
        obs, _, done, info = env.step(action=actions)
    all_obs.append(obs)
    env.close()

    labels = list(parse_pickandplace_environment_observations(observations=obs).keys())
    df = pd.DataFrame(data=all_obs, columns=labels)

    df.to_csv(path_or_buf=output_path, sep=' ', index=False)

    return df


def visualise_episode(
        specification: RobotSpecification
        ) -> None:
    robot = ManipulatorCalibrationValidationRobot(specification=specification)
    env = get_environment_config().environment(robot=robot, wrap2gym=False)

    def policy_fn(
            timestep: TimeStep
            ):
        obs = _flatten_obs(timestep.observation)
        actions = robot(obs)
        return actions

    viewer.launch(env, policy_fn)
    env.close()


if __name__ == '__main__':
    labels = ["NEA", "PEA", "BPEA", "FULL"]
    specifications = [nea_specification(), pea_specification(), bea_specification(), full_specification()]

    visualise_episode(pea_specification())

    base_path = Path(f"./calibration/pap_validation/data/sim")
    for label, specification in zip(labels, specifications):
        path = base_path / label
        path.mkdir(exist_ok=True, parents=True)

        df = record_episode_statistics(specification=specification, output_path=str(path / "raw.csv"))

        # split df into shoulder and elbow
        # Split shoulder and elbow into go and return
        joints = ["shoulder", "elbow"]
        phases = ["go", "return"]
        phases_time_ranges = [(4, 7), (9, 12)]
        joint_attributes = ["q", "q_vel", "torque"]
        for joint in joints:
            for phase, time_range in zip(phases, phases_time_ranges):
                selected_df = df[["timestep"] + [f"{joint}_{attribute}" for attribute in joint_attributes]]
                selected_df = selected_df[
                    (time_range[0] <= selected_df["timestep"]) & (selected_df["timestep"] <= time_range[1])]
                selected_df["timestep"] -= time_range[0]

                selected_df.to_csv(path_or_buf=str(path / f"{joint}_{phase}.csv"), sep=' ', index=False)
