from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Optional, List

import numpy as np

from elastic_actuation_arm.calibration.optimization.robot.controller import \
    ManipulatorCalibrationControllerSpecification
from elastic_actuation_arm.calibration.optimization.robot.specification import ManipulatorCalibrationSpecification
from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from erpy.base.genome import ESGenome, ESGenomeConfig, Genome
from erpy.base.parameters import ContinuousParameter


@dataclass
class ManipulatorCalibrationGenomeConfig(ESGenomeConfig):
    @property
    def genome(self) -> Type[ManipulatorCalibrationGenome]:
        return ManipulatorCalibrationGenome

    def extract_parameters(self, specification: ManipulatorCalibrationSpecification) -> List[ContinuousParameter]:
        morph_spec = specification.morphology_specification
        parameters = [morph_spec.upper_arm_spec.joint_damping,
                      morph_spec.upper_arm_spec.joint_friction_loss,
                      morph_spec.upper_arm_spec.joint_armature,
                      morph_spec.fore_arm_spec.joint_damping,
                      morph_spec.fore_arm_spec.joint_friction_loss,
                      morph_spec.fore_arm_spec.joint_armature,
                      ]
        return parameters

    def base_specification(self) -> ManipulatorCalibrationSpecification:
        morph_spec = ManipulatorMorphologySpecification.default()
        morph_spec.end_effector_spec.adhesion.value = False
        control_spec = ManipulatorCalibrationControllerSpecification.default()

        return ManipulatorCalibrationSpecification(morphology_specification=morph_spec,
                                                   controller_specification=control_spec)

    @property
    def parameter_labels(self) -> List[str]:
        return ["shoulder_joint_damping", "shoulder_friction_loss", "shoulder_armature",
                "elbow_joint_damping", "elbow_friction_loss", "elbow_armature"]


class ManipulatorCalibrationGenome(ESGenome):
    def __init__(self, parameters: np.ndarray,
                 config: ManipulatorCalibrationGenomeConfig,
                 genome_id: int, parent_genome_id: Optional[int] = None):
        super().__init__(parameters=parameters, config=config, genome_id=genome_id, parent_genome_id=parent_genome_id)

    @property
    def config(self) -> ManipulatorCalibrationGenomeConfig:
        return super().config

    @staticmethod
    def generate(config: ManipulatorCalibrationGenomeConfig,
                 genome_id: int, *args, **kwargs) -> ManipulatorCalibrationGenome:
        parameters = config.random_state.rand(config.num_parameters)
        return ManipulatorCalibrationGenome(parameters=parameters, config=config, genome_id=genome_id)

    @property
    def specification(self) -> ManipulatorCalibrationSpecification:
        return super().specification

    def mutate(self, child_genome_id: int, *args, **kwargs) -> ESGenome:
        pass

    def cross_over(self, partner_genome: Genome, child_genome_id: int) -> ESGenome:
        pass
