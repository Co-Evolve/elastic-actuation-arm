from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from erpy.base.parameters import FixedParameter, ContinuousParameter
from erpy.base.specification import MorphologySpecification, Specification


class PActuatorSpecification(Specification):
    def __init__(self, gear: float, kp: float, force_limit=Tuple[float, float]) -> None:
        self.gear = FixedParameter(value=gear)
        self.kp = FixedParameter(value=kp)
        self.force_limit = FixedParameter(value=force_limit)


class ParallelSpringSpecification(Specification):
    def __init__(self, stiffness: ContinuousParameter, equilibrium_angle: ContinuousParameter) -> None:
        self.stiffness = stiffness
        self.equilibrium_angle = equilibrium_angle


class BiarticularSpringSpecification(Specification):
    def __init__(self, stiffness: float, q0: float, r1: float, r2: float, r3: float) -> None:
        spr1 = {i * 2.057 for i in range(6)}
        spr2 = {i * 3.426 for i in range(5)}
        stiffness_options = list(spr1.union(spr2))

        r1_r3_options = [0.013, 0.0135, 0.014, 0.018, 0.0185, 0.019, 0.0205, 0.021, 0.022, 0.028, 0.0285, 0.038]
        r2_options = [0.018, 0.021, 0.022, 0.0225, 0.025, 0.026, 0.027, 0.035]

        self.stiffness = ContinuousParameter(low=0, high=max(stiffness_options), value=stiffness)
        self.q0 = ContinuousParameter(low=-np.pi, high=np.pi, value=q0)
        self.r1 = ContinuousParameter(low=min(r1_r3_options), high=max(r1_r3_options), value=r1)
        self.r2 = ContinuousParameter(low=min(r2_options), high=max(r2_options), value=r2)
        self.r3 = ContinuousParameter(low=min(r1_r3_options), high=max(r1_r3_options), value=r3)


class LinkSpecification(Specification):
    def __init__(self, joint_range: List[float], joint_damping: float, joint_armature: float,
                 joint_friction_loss: float, mass: float,
                 p_actuator_spec: PActuatorSpecification, spring_spec: ParallelSpringSpecification) -> None:
        self.joint_damping = FixedParameter(value=joint_damping)
        self.joint_friction_loss = FixedParameter(value=joint_friction_loss)
        self.joint_armature = FixedParameter(value=joint_armature)

        self.joint_range = FixedParameter(value=joint_range)
        self.mass = FixedParameter(value=mass)
        self.p_actuator_spec = p_actuator_spec
        self.spring_spec = spring_spec


class EndEffectorSpecification(Specification):
    def __init__(self, adhesion: bool) -> None:
        self.adhesion = FixedParameter(adhesion)


@dataclass
class ManipulatorMorphologySpecification(MorphologySpecification):
    base_spec: LinkSpecification
    upper_arm_spec: LinkSpecification
    fore_arm_spec: LinkSpecification
    end_effector_spec: EndEffectorSpecification
    biarticular_spring_spec: BiarticularSpringSpecification

    @property
    def joint_ranges(self) -> np.array:
        return np.vstack([
            self.base_spec.joint_range.value,
            self.upper_arm_spec.joint_range.value,
            self.fore_arm_spec.joint_range.value
        ]).T

    @staticmethod
    def default() -> ManipulatorMorphologySpecification:
        base_spec = LinkSpecification(
            joint_range=np.array([-np.pi, np.pi]),
            joint_damping=2,
            joint_armature=3.189,
            joint_friction_loss=9.006,
            mass=4.524,
            p_actuator_spec=None,
            spring_spec=None
        )
        upper_arm_spec = LinkSpecification(
            joint_range=np.array([-10 / 180 * np.pi, 150 / 180 * np.pi]),
            joint_damping=39.958,
            joint_armature=0.2254,
            joint_friction_loss=7.877,
            mass=4.029,
            p_actuator_spec=PActuatorSpecification(gear=1, kp=572.9577951308232, force_limit=[-157.5, 157.5]),
            spring_spec=ParallelSpringSpecification(
                stiffness=ContinuousParameter(low=0.0, high=13.705, value=13.705),
                equilibrium_angle=ContinuousParameter(low=-10 / 180 * np.pi,
                                                      high=140 / 180 * np.pi,
                                                      value=110 / 180 * np.pi))
        )

        fore_arm_spec = LinkSpecification(
            joint_range=np.array([-140 / 180 * np.pi, 140 / 180 * np.pi]),
            joint_damping=22.831,
            joint_armature=0.03064,
            joint_friction_loss=6.128,
            mass=2.181,
            p_actuator_spec=PActuatorSpecification(gear=1, kp=572.9577951308232, force_limit=[-95.25, 95.25]),
            spring_spec=ParallelSpringSpecification(stiffness=ContinuousParameter(low=0.0, high=1.32,
                                                                                  value=1.32),
                                                    equilibrium_angle=ContinuousParameter(low=-140 / 180 * np.pi,
                                                                                          high=140 / 180 * np.pi,
                                                                                          value=-140 / 180 * np.pi))
        )

        end_effector_spec = EndEffectorSpecification(adhesion=True)

        biarticular_spring_spec = BiarticularSpringSpecification(stiffness=3.426,
                                                                 q0=-150 / 180 * np.pi,
                                                                 r1=0.019,
                                                                 r2=0.018,
                                                                 r3=0.014)

        return ManipulatorMorphologySpecification(base_spec=base_spec,
                                                  upper_arm_spec=upper_arm_spec,
                                                  fore_arm_spec=fore_arm_spec,
                                                  end_effector_spec=end_effector_spec,
                                                  biarticular_spring_spec=biarticular_spring_spec)

    @staticmethod
    def pea_default() -> ManipulatorMorphologySpecification:
        specification = ManipulatorMorphologySpecification.default()
        specification.biarticular_spring_spec.stiffness = FixedParameter(0.0)
        return specification

    @staticmethod
    def bea_default() -> ManipulatorMorphologySpecification:
        specification = ManipulatorMorphologySpecification.default()
        specification.upper_arm_spec.spring_spec.stiffness = FixedParameter(0.0)
        specification.fore_arm_spec.spring_spec.stiffness = FixedParameter(0.0)
        return specification

    @staticmethod
    def nea_default() -> ManipulatorMorphologySpecification:
        specification = ManipulatorMorphologySpecification.default()
        specification.biarticular_spring_spec.stiffness = FixedParameter(0.0)
        specification.upper_arm_spec.spring_spec.stiffness = FixedParameter(0.0)
        specification.fore_arm_spec.spring_spec.stiffness = FixedParameter(0.0)
        return specification

    @staticmethod
    def real_pea() -> ManipulatorMorphologySpecification:
        specification = ManipulatorMorphologySpecification.pea_default()
        specification.upper_arm_spec.spring_spec.stiffness = FixedParameter(13.7052)
        specification.fore_arm_spec.spring_spec.stiffness = FixedParameter(0.8755)
        return specification

    @staticmethod
    def real_bea() -> ManipulatorMorphologySpecification:
        specification = ManipulatorMorphologySpecification.bea_default()
        specification.biarticular_spring_spec.stiffness = FixedParameter(10.9664)
        specification.biarticular_spring_spec.r1 = FixedParameter(0.014)
        specification.biarticular_spring_spec.r2 = FixedParameter(0.035)
        specification.biarticular_spring_spec.r3 = FixedParameter(0.013)
        return specification

    @staticmethod
    def real_full() -> ManipulatorMorphologySpecification:
        specification = ManipulatorMorphologySpecification.default()
        specification.upper_arm_spec.spring_spec.stiffness = FixedParameter(8.2276)
        specification.fore_arm_spec.spring_spec.stiffness = FixedParameter(0.6589)

        specification.biarticular_spring_spec.stiffness = FixedParameter(13.7052)
        specification.biarticular_spring_spec.r1 = FixedParameter(0.013)
        specification.biarticular_spring_spec.r2 = FixedParameter(0.035)
        specification.biarticular_spring_spec.r3 = FixedParameter(0.018)
        return specification

    @staticmethod
    def real_full_v1() -> ManipulatorMorphologySpecification:
        specification = ManipulatorMorphologySpecification.default()
        specification.upper_arm_spec.spring_spec.stiffness = FixedParameter(8.2276)
        specification.fore_arm_spec.spring_spec.stiffness = FixedParameter(0.6589)

        specification.upper_arm_spec.spring_spec.equilibrium_angle = FixedParameter(107.7522 / 180 * np.pi)
        specification.fore_arm_spec.spring_spec.equilibrium_angle = FixedParameter(-62.4224 / 180 * np.pi)

        specification.biarticular_spring_spec.stiffness = FixedParameter(13.7052)
        specification.biarticular_spring_spec.r1 = FixedParameter(0.013)
        specification.biarticular_spring_spec.r2 = FixedParameter(0.035)
        specification.biarticular_spring_spec.r3 = FixedParameter(0.018)
        return specification
