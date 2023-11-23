from __future__ import annotations

import numpy as np

from elastic_actuation_arm.pick_and_place.environment.environment import parse_pickandplace_environment_observations
from erpy.base.parameters import FixedParameter
from erpy.base.phenome import Controller
from erpy.base.specification import ControllerSpecification


class ManipulatorCalibrationValidationControllerSpecification(ControllerSpecification):
    def __init__(
            self,
            start_q: np.ndarray,
            end_q: np.ndarray
            ) -> None:
        self.start_q = FixedParameter(start_q)
        self.end_q = FixedParameter(end_q)


class ManipulatorCalibrationValidationController(Controller):
    def __init__(
            self,
            specification: ManipulatorCalibrationValidationControllerSpecification
            ) -> None:
        super().__init__(specification)

    @property
    def specification(
            self
            ) -> ManipulatorCalibrationValidationControllerSpecification:
        return super().specification

    @staticmethod
    def interpolate(
            q0: np.ndarray,
            qf: np.ndarray,
            tf: float,
            t: float
            ) -> np.ndarray:
        return q0 + ((qf - q0) / np.power(tf, 3)) * (
                ((6 / np.power(tf, 2)) * np.power(t, 5))
                - ((15 / tf) * np.power(t, 4))
                + (10 * np.power(t, 3))
        )

    def __call__(
            self,
            observations: np.ndarray
            ) -> np.ndarray:
        obs_dict = parse_pickandplace_environment_observations(observations=observations)
        timestep = obs_dict['timestep']
        initialisation_duration = obs_dict['initialisation_duration']
        placement_duration = obs_dict['placement_duration']
        go_duration = obs_dict['go_duration']
        ret_duration = obs_dict['ret_duration']

        if timestep < initialisation_duration:
            # INITIALISATION PHASE -> GO TO START WITH PAYLOAD
            radians = self.specification.start_q.value

            # Go to target more slowly during initialisation
            radians = radians * min(1, timestep / (0.5 * initialisation_duration))

            adhesion = 1.0
        else:
            relative_timestep = timestep - initialisation_duration

            if relative_timestep < go_duration:
                # GO PHASE
                adhesion = 1.0
                radians = self.interpolate(
                        q0=self.specification.start_q.value,
                        qf=self.specification.end_q.value,
                        tf=go_duration,
                        t=relative_timestep
                        )
                # radians = self.go_interpolator(relative_timestep)
                # direction = self.specification.end_q.value - self.specification.start_q.value
                # radians = self.specification.start_q.value + (relative_timestep / go_duration) * direction
            else:
                relative_timestep -= go_duration
                if relative_timestep < placement_duration:
                    # PLACEMENT PHASE
                    relative_timestep = min(relative_timestep, go_duration)
                    adhesion = max(0, 1 - relative_timestep / (placement_duration / 2))
                    # radians = self.go_interpolator(self.specification.go_duration.value)
                    radians = self.specification.end_q.value
                else:
                    # RET PHASE
                    relative_timestep -= placement_duration
                    relative_timestep = min(relative_timestep, ret_duration)

                    adhesion = 0.0
                    # radians = self.ret_interpolator(relative_timestep)

                    radians = self.interpolate(
                            q0=self.specification.end_q.value,
                            qf=self.specification.start_q.value,
                            tf=ret_duration,
                            t=relative_timestep
                            )
                    # direction = self.specification.start_q.value - self.specification.end_q.value
                    # radians = self.specification.end_q.value + (relative_timestep / ret_duration) * direction

        # [payload, shoulder, elbow, end-effector]
        return np.concatenate(([1 - adhesion], radians[1:], [adhesion]))
