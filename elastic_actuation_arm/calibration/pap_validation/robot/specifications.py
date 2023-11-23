import numpy as np

from elastic_actuation_arm.calibration.pap_validation.robot.controller import \
    ManipulatorCalibrationValidationControllerSpecification
from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from erpy.base.parameters import FixedParameter
from erpy.base.specification import RobotSpecification


def nea_specification() -> RobotSpecification:
    start_q = np.array([0.0, 69.250366, 138.560181]) / 180 * np.pi
    end_q = np.array([0.0, 19.480927, 38.951263]) / 180 * np.pi

    morphology_specification = ManipulatorMorphologySpecification.nea_default()
    controller_specification = ManipulatorCalibrationValidationControllerSpecification(
            start_q=start_q, end_q=end_q
            )
    specification = RobotSpecification(
            morphology_specification=morphology_specification, controller_specification=controller_specification
            )
    return specification


def pea_specification() -> RobotSpecification:
    start_q = np.array([0.0, 69.652222, 138.777542]) / 180 * np.pi
    end_q = np.array([0.0, 19.480408, 38.947525]) / 180 * np.pi

    morphology_specification = ManipulatorMorphologySpecification.nea_default()
    morphology_specification.upper_arm_spec.spring_spec.stiffness = FixedParameter(13.7052)
    morphology_specification.upper_arm_spec.spring_spec.equilibrium_angle = FixedParameter(110 / 180 * np.pi)
    morphology_specification.fore_arm_spec.spring_spec.stiffness = FixedParameter(1.318)
    morphology_specification.fore_arm_spec.spring_spec.equilibrium_angle = FixedParameter(-140 / 180 * np.pi)

    controller_specification = ManipulatorCalibrationValidationControllerSpecification(
            start_q=start_q, end_q=end_q
            )
    specification = RobotSpecification(
            morphology_specification=morphology_specification, controller_specification=controller_specification
            )
    return specification


def bea_specification() -> RobotSpecification:
    start_q = np.array([0.0, 69.520645, 138.513992]) / 180 * np.pi
    end_q = np.array([0.0, 19.480671, 38.947647]) / 180 * np.pi

    morphology_specification = ManipulatorMorphologySpecification.nea_default()
    morphology_specification.biarticular_spring_spec.stiffness = FixedParameter(3.425)
    morphology_specification.biarticular_spring_spec.q0 = FixedParameter(-150 / 180 * np.pi)
    morphology_specification.biarticular_spring_spec.r1 = FixedParameter(0.019)
    morphology_specification.biarticular_spring_spec.r2 = FixedParameter(0.019)
    morphology_specification.biarticular_spring_spec.r3 = FixedParameter(0.014)

    controller_specification = ManipulatorCalibrationValidationControllerSpecification(
            start_q=start_q, end_q=end_q
            )
    specification = RobotSpecification(
            morphology_specification=morphology_specification, controller_specification=controller_specification
            )
    return specification


def full_specification() -> RobotSpecification:
    start_q = np.array([0.0, 69.520645, 138.513992]) / 180 * np.pi
    end_q = np.array([0.0, 19.480671, 38.947647]) / 180 * np.pi

    morphology_specification = ManipulatorMorphologySpecification.nea_default()
    morphology_specification.upper_arm_spec.spring_spec.stiffness = FixedParameter(13.7052)
    morphology_specification.upper_arm_spec.spring_spec.equilibrium_angle = FixedParameter(100 / 180 * np.pi)
    morphology_specification.fore_arm_spec.spring_spec.stiffness = FixedParameter(0)
    morphology_specification.fore_arm_spec.spring_spec.equilibrium_angle = FixedParameter(0 / 180 * np.pi)
    morphology_specification.biarticular_spring_spec.stiffness = FixedParameter(4.40)
    morphology_specification.biarticular_spring_spec.q0 = FixedParameter(-180 / 180 * np.pi)
    morphology_specification.biarticular_spring_spec.r1 = FixedParameter(0.019)
    morphology_specification.biarticular_spring_spec.r2 = FixedParameter(0.019)
    morphology_specification.biarticular_spring_spec.r3 = FixedParameter(0.014)

    controller_specification = ManipulatorCalibrationValidationControllerSpecification(
            start_q=start_q, end_q=end_q
            )
    specification = RobotSpecification(
            morphology_specification=morphology_specification, controller_specification=controller_specification
            )
    return specification
