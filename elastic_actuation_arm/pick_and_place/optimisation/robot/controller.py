from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from elastic_actuation_arm.pick_and_place.environment.environment import \
    parse_pickandplace_environment_observations
from erpy.base.parameters import FixedParameter, ContinuousParameter
from erpy.base.phenome import Controller
from erpy.base.specification import ControllerSpecification

NUM_INTERMEDIATE_POINTS = 5
START_Q = np.array([0.78539816, 1.2094294, 2.41885879])
END_Q = np.array([-0.78539816, 0.33983655, 0.6796731])


@dataclass
class ManipulatorAkimaSpineControllerSpecification(ControllerSpecification):
    go_intermediate_q: np.ndarray[ContinuousParameter]
    ret_intermediate_q: np.ndarray[ContinuousParameter]
    go_duration: ContinuousParameter
    ret_duration: ContinuousParameter

    start_q: np.ndarray[FixedParameter] = np.array([FixedParameter(q) for q in START_Q])
    end_q: np.ndarray[FixedParameter] = np.array([FixedParameter(q) for q in END_Q])

    _all_q: np.ndarray = None
    _go_q: np.ndarray = None
    _ret_q: np.ndarray = None

    @staticmethod
    def default(morphology_spec: ManipulatorMorphologySpecification) -> ManipulatorAkimaSpineControllerSpecification:
        time = np.linspace(0, 1, NUM_INTERMEDIATE_POINTS + 2)[1:-1].reshape(NUM_INTERMEDIATE_POINTS, 1)
        go_intermediate_q = START_Q + time * (END_Q - START_Q)
        ret_intermediate_q = go_intermediate_q[::-1]

        # Set parameter ranges for intermediate points
        #   Give it 1.5 the range required to complete the trajectory
        start_end_qs = np.vstack((START_Q, END_Q))
        low_qs = np.min(start_end_qs, axis=0)
        high_qs = np.max(start_end_qs, axis=0)
        current_range = np.abs(high_qs - low_qs)
        stretch = current_range / 4

        low_qs -= stretch
        high_qs += stretch
        #   But make sure that it is still in a valid joint range
        min_joint_q, max_joint_q = morphology_spec.joint_ranges
        min_joint_q += 0.01
        max_joint_q -= 0.01

        low_qs = low_qs.clip(min=min_joint_q, max=max_joint_q)
        high_qs = high_qs.clip(min=min_joint_q, max=max_joint_q)

        #   Create parameters
        def create_param(qs: np.ndarray[float]) -> np.ndarray[ContinuousParameter]:
            return np.array([ContinuousParameter(low=low_qs[i],
                                                 high=high_qs[i],
                                                 value=qs[i]) for i in range(3)])

        go_intermediate_q = np.apply_along_axis(create_param, 1, go_intermediate_q)
        ret_intermediate_q = np.apply_along_axis(create_param, 1, ret_intermediate_q)

        return ManipulatorAkimaSpineControllerSpecification(go_intermediate_q=go_intermediate_q,
                                                            ret_intermediate_q=ret_intermediate_q,
                                                            go_duration=ContinuousParameter(low=1., high=5., value=2.5),
                                                            ret_duration=ContinuousParameter(low=1., high=5.,
                                                                                             value=2.5))

    @property
    def num_points(self) -> int:
        return self.all_q.shape[0]

    @property
    def all_q(self) -> np.ndarray:
        if self._all_q is None:
            params = np.vstack(
                (self.start_q, self.go_intermediate_q, self.end_q, self.ret_intermediate_q, self.start_q))
            self._all_q = np.vectorize(lambda param: param.value)(params)

        return self._all_q

    @property
    def go_q(self) -> np.ndarray:
        if self._go_q is None:
            params = np.vstack(
                (self.start_q, self.go_intermediate_q, self.end_q))
            self._go_q = np.vectorize(lambda param: param.value)(params)
        return self._go_q

    @property
    def ret_q(self) -> np.ndarray:
        if self._ret_q is None:
            params = np.vstack(
                (self.end_q, self.ret_intermediate_q, self.start_q))
            self._ret_q = np.vectorize(lambda param: param.value)(params)
        return self._ret_q


class ManipulatorAkimaSpineController(Controller):
    def __init__(self, specification: ManipulatorAkimaSpineControllerSpecification):
        super().__init__(specification)
        self._go_interpolator = None
        self._ret_interpolator = None

    @property
    def specification(self) -> ManipulatorAkimaSpineControllerSpecification:
        return super().specification

    @property
    def go_interpolator(self):
        if self._go_interpolator is None:
            y = self.specification.go_q
            x = np.linspace(start=0,
                            stop=self.specification.go_duration.value,
                            num=self.specification.go_q.shape[0])
            self._go_interpolator = Akima1DInterpolator(x=x, y=y)
        return self._go_interpolator

    @property
    def ret_interpolator(self):
        if self._ret_interpolator is None:
            y = self.specification.ret_q
            x = np.linspace(start=0,
                            stop=self.specification.ret_duration.value,
                            num=self.specification.ret_q.shape[0])
            self._ret_interpolator = Akima1DInterpolator(x=x, y=y)
        return self._ret_interpolator

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        obs_dict = parse_pickandplace_environment_observations(observations=observations)
        timestep = obs_dict['timestep']
        initialisation_duration = obs_dict['initialisation_duration']
        placement_duration = obs_dict['placement_duration']

        if timestep < initialisation_duration:
            # INITIALISATION PHASE -> GO TO START WITH PAYLOAD
            relative_timestep = 0.0

            radians = self.go_interpolator(relative_timestep)

            # Go to target more slowly during initialisation
            radians *= min(1, timestep / (0.5 * initialisation_duration))

            adhesion = 1.0
        else:
            relative_timestep = timestep - initialisation_duration

            if relative_timestep < self.specification.go_duration.value:
                # GO PHASE
                adhesion = 1.0
                radians = self.go_interpolator(relative_timestep)
            else:
                relative_timestep -= self.specification.go_duration.value
                if relative_timestep < placement_duration:
                    # PLACEMENT PHASE
                    adhesion = max(0, 1 - relative_timestep / (placement_duration / 2))
                    radians = self.go_interpolator(self.specification.go_duration.value)
                else:
                    # RET PHASE
                    relative_timestep -= placement_duration
                    relative_timestep = min(relative_timestep, self.specification.ret_duration.value)
                    adhesion = 0.0
                    radians = self.ret_interpolator(relative_timestep)

        # [payload, shoulder, elbow, end-effector]
        return np.concatenate(([1 - adhesion], radians[1:], [adhesion]))


if __name__ == '__main__':
    spec = ManipulatorAkimaSpineControllerSpecification.default(ManipulatorMorphologySpecification.default())
    controller = ManipulatorAkimaSpineController(spec)

    times = np.linspace(0, spec.ret_duration.value, 10)
    shoulder_qs = []
    elbow_qs = []
    for t in times:
        _, shoulder_q, elbow_q = controller.ret_interpolator(t)
        shoulder_qs.append(shoulder_q)
        elbow_qs.append(elbow_q)

    import matplotlib.pyplot as plt

    plt.plot(times, shoulder_qs)
    plt.xlabel('time')
    plt.ylabel('q')
    plt.title('shoulder')
    plt.show()
    plt.close()
    plt.plot(times, elbow_qs)
    plt.xlabel('time')
    plt.ylabel('q')
    plt.title('elbow')
    plt.show()
    plt.close()
