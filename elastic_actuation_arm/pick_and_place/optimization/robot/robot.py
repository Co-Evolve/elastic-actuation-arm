from elastic_actuation_arm.entities.manipulator.manipulator import ManipulatorMorphology
from elastic_actuation_arm.pick_and_place.optimization.robot.controller import \
    ManipulatorAkimaSpineController
from elastic_actuation_arm.pick_and_place.optimization.robot.specification import \
    ManipulatorPickAndPlaceSpringTrajectorySpecification
from erpy.interfaces.mujoco.phenome import MJCRobot


class ManipulatorPickAndPlaceSpringTrajectoryRobot(MJCRobot):
    def __init__(self, specification: ManipulatorPickAndPlaceSpringTrajectorySpecification):
        super().__init__(specification=specification)

    @property
    def morphology(self) -> ManipulatorMorphology:
        return super().morphology

    @property
    def controller(self) -> ManipulatorAkimaSpineController:
        return super().controller

    def reset(self) -> None:
        pass

    def _build(self) -> None:
        self._morphology = ManipulatorMorphology(specification=self.specification.morphology_specification)
        self._controller = ManipulatorAkimaSpineController(
            specification=self.specification.controller_specification)
