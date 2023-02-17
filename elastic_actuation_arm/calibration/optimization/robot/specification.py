from __future__ import annotations

from dataclasses import dataclass

from elastic_actuation_arm.calibration.optimization.robot.controller import \
    ManipulatorCalibrationControllerSpecification
from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from erpy.base.specification import RobotSpecification


@dataclass
class ManipulatorCalibrationSpecification(RobotSpecification):
    morphology_specification: ManipulatorMorphologySpecification
    controller_specification: ManipulatorCalibrationControllerSpecification

    @staticmethod
    def default() -> ManipulatorCalibrationSpecification:
        return ManipulatorCalibrationSpecification(
            morphology_specification=ManipulatorMorphologySpecification.pea_default(),
            controller_specification=ManipulatorCalibrationControllerSpecification.default())
