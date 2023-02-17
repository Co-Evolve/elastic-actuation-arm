from __future__ import annotations

from dataclasses import dataclass

from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from elastic_actuation_arm.pick_and_place.optimization.robot.controller import \
    ManipulatorAkimaSpineControllerSpecification
from erpy.base.specification import RobotSpecification


@dataclass
class ManipulatorPickAndPlaceSpringTrajectorySpecification(RobotSpecification):
    morphology_specification: ManipulatorMorphologySpecification
    controller_specification: ManipulatorAkimaSpineControllerSpecification

    @staticmethod
    def default() -> ManipulatorPickAndPlaceSpringTrajectorySpecification:
        morph_spec = ManipulatorMorphologySpecification.default()
        controller_spec = ManipulatorAkimaSpineControllerSpecification.default(morphology_spec=morph_spec)
        return ManipulatorPickAndPlaceSpringTrajectorySpecification(
            morphology_specification=morph_spec,
            controller_specification=controller_spec)

    @staticmethod
    def nea_default() -> ManipulatorPickAndPlaceSpringTrajectorySpecification:
        morph_spec = ManipulatorMorphologySpecification.nea_default()
        controller_spec = ManipulatorAkimaSpineControllerSpecification.default(morphology_spec=morph_spec)
        return ManipulatorPickAndPlaceSpringTrajectorySpecification(
            morphology_specification=morph_spec,
            controller_specification=controller_spec)

    @staticmethod
    def pea_default() -> ManipulatorPickAndPlaceSpringTrajectorySpecification:
        morph_spec = ManipulatorMorphologySpecification.pea_default()
        controller_spec = ManipulatorAkimaSpineControllerSpecification.default(morphology_spec=morph_spec)
        return ManipulatorPickAndPlaceSpringTrajectorySpecification(
            morphology_specification=morph_spec,
            controller_specification=controller_spec)

    @staticmethod
    def bea_default() -> ManipulatorPickAndPlaceSpringTrajectorySpecification:
        morph_spec = ManipulatorMorphologySpecification.bea_default()
        controller_spec = ManipulatorAkimaSpineControllerSpecification.default(morphology_spec=morph_spec)
        return ManipulatorPickAndPlaceSpringTrajectorySpecification(
            morphology_specification=morph_spec,
            controller_specification=controller_spec)

    @staticmethod
    def real_pea_default() -> ManipulatorPickAndPlaceSpringTrajectorySpecification:
        morph_spec = ManipulatorMorphologySpecification.real_pea()
        controller_spec = ManipulatorAkimaSpineControllerSpecification.default(morphology_spec=morph_spec)
        return ManipulatorPickAndPlaceSpringTrajectorySpecification(
            morphology_specification=morph_spec,
            controller_specification=controller_spec)

    @staticmethod
    def real_bea_default() -> ManipulatorPickAndPlaceSpringTrajectorySpecification:
        morph_spec = ManipulatorMorphologySpecification.real_bea()
        controller_spec = ManipulatorAkimaSpineControllerSpecification.default(morphology_spec=morph_spec)
        return ManipulatorPickAndPlaceSpringTrajectorySpecification(
            morphology_specification=morph_spec,
            controller_specification=controller_spec)

    @staticmethod
    def real_full_default() -> ManipulatorPickAndPlaceSpringTrajectorySpecification:
        morph_spec = ManipulatorMorphologySpecification.real_full()
        controller_spec = ManipulatorAkimaSpineControllerSpecification.default(morphology_spec=morph_spec)
        return ManipulatorPickAndPlaceSpringTrajectorySpecification(
            morphology_specification=morph_spec,
            controller_specification=controller_spec)

    @staticmethod
    def real_full_v1_default() -> ManipulatorPickAndPlaceSpringTrajectorySpecification:
        morph_spec = ManipulatorMorphologySpecification.real_full_v1()
        controller_spec = ManipulatorAkimaSpineControllerSpecification.default(morphology_spec=morph_spec)
        return ManipulatorPickAndPlaceSpringTrajectorySpecification(
            morphology_specification=morph_spec,
            controller_specification=controller_spec)
