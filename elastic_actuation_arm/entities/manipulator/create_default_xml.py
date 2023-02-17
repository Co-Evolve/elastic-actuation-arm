from dm_control.mjcf import export_with_assets

from elastic_actuation_arm.entities.manipulator.manipulator import ManipulatorMorphology
from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification

if __name__ == '__main__':
    morphology = ManipulatorMorphology(specification=ManipulatorMorphologySpecification.default())

    export_with_assets(morphology.mjcf_model, 'robot_xml')
