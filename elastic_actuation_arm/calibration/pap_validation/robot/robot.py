from elastic_actuation_arm.calibration.pap_validation.robot.controller import ManipulatorCalibrationValidationController
from elastic_actuation_arm.entities.manipulator.manipulator import ManipulatorMorphology
from erpy.base.specification import RobotSpecification
from erpy.interfaces.mujoco.phenome import MJCRobot


class ManipulatorCalibrationValidationRobot(MJCRobot):
    def __init__(
            self,
            specification: RobotSpecification
            ):
        super().__init__(specification=specification)

    @property
    def morphology(
            self
            ) -> ManipulatorMorphology:
        return super().morphology

    @property
    def controller(
            self
            ) -> ManipulatorCalibrationValidationController:
        return super().controller

    def reset(
            self
            ) -> None:
        pass

    def _build(
            self
            ) -> None:
        self._morphology = ManipulatorMorphology(specification=self.specification.morphology_specification)
        self._controller = ManipulatorCalibrationValidationController(
            specification=self.specification.controller_specification
            )
