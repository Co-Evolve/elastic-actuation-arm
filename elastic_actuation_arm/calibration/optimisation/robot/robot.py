from elastic_actuation_arm.calibration.optimisation.robot.controller import ManipulatorCalibrationController
from elastic_actuation_arm.entities.manipulator.manipulator import ManipulatorMorphology
from erpy.base.specification import RobotSpecification
from erpy.interfaces.mujoco.phenome import MJCRobot


class ManipulatorCalibrationRobot(MJCRobot):
    def __init__(self, specification: RobotSpecification):
        super().__init__(specification=specification)

    @property
    def morphology(self) -> ManipulatorMorphology:
        return super().morphology

    @property
    def controller(self) -> ManipulatorCalibrationController:
        return super().controller

    def reset(self) -> None:
        pass

    def _build(self) -> None:
        self._morphology = ManipulatorMorphology(specification=self.specification.morphology_specification)
        self._controller = ManipulatorCalibrationController(specification=self.specification.controller_specification)
