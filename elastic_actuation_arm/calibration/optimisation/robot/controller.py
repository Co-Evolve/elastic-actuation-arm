from __future__ import annotations

import numpy as np

from elastic_actuation_arm.calibration.environment.environment import \
    parse_calibration_environment_observations
from erpy.base.parameters import FixedParameter
from erpy.base.phenome import Controller
from erpy.base.specification import ControllerSpecification


class ManipulatorCalibrationControllerSpecification(ControllerSpecification):
    def __init__(self, f0, f1, amp) -> None:
        self.f0 = FixedParameter(f0)
        self.f1 = FixedParameter(f1)
        self.amp = FixedParameter(amp)

    @staticmethod
    def default() -> ManipulatorCalibrationControllerSpecification:
        f0 = np.array([0.2, 0.2, 0.1])
        f1 = np.array([0.05, 0.01, 0.01])
        amp = np.array([45, 45, 135])
        return ManipulatorCalibrationControllerSpecification(f0=f0,
                                                             f1=f1,
                                                             amp=amp)


class ManipulatorCalibrationController(Controller):
    def __init__(self, specification: ManipulatorCalibrationControllerSpecification):
        super().__init__(specification)
        self.high_to_low = False

    @property
    def specification(self) -> ManipulatorCalibrationControllerSpecification:
        return super().specification

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        # Return the next target joint angles from the given sequence
        obs_dict = parse_calibration_environment_observations(observations=observations)
        q0 = np.array([0.0, obs_dict['shoulder_q0'], obs_dict['elbow_q0']])
        high_to_low = obs_dict['high_to_low']
        time = obs_dict['timestep']

        if high_to_low:
            f0 = self.specification.f0.value
            f1 = self.specification.f1.value
        else:
            f0 = self.specification.f1.value
            f1 = self.specification.f0.value

        degrees = self.specification.amp.value * time / 60 * np.sin(
            2 * np.pi * (f0 * time + (f1 - f0) / (2 * 60) * (time ** 2)))
        radians = degrees / 180 * np.pi
        radians += q0
        return radians[1:]
