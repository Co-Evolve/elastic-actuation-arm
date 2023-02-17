import numpy as np
from dm_control.mjcf import export_with_assets

from elastic_actuation_arm.entities.manipulator.elastic_actuation import apply_elastic_actuation
from elastic_actuation_arm.entities.manipulator.parts.base import Base
from elastic_actuation_arm.entities.manipulator.parts.end_effector import EndEffector
from elastic_actuation_arm.entities.manipulator.parts.forearm import ForeArm
from elastic_actuation_arm.entities.manipulator.parts.upperarm import UpperArm
from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from erpy.interfaces.mujoco.phenome import MJCMorphology, MJCMorphologyPart


class ManipulatorMorphology(MJCMorphology):
    def __init__(self, specification: ManipulatorMorphologySpecification):
        super().__init__(specification, model_name='elastic-actuation-arm')

    @property
    def specification(self) -> ManipulatorMorphologySpecification:
        return super().specification

    def _build(self):
        self._configure_compiler()
        self._configure_assets()
        self._configure_parts()

    def _configure_compiler(self) -> None:
        self.mjcf_model.compiler.discardvisual = True

        self.mjcf_model.compiler.angle = 'radian'

    def _configure_assets(self) -> None:
        self.mjcf_model.compiler.meshdir = "entities/stl_files"
        mesh_scale = [0.001, 0.001, 0.001]

        self.mjcf_model.asset.add('mesh', name='base', file='manipulator/link0.stl', scale=mesh_scale)
        self.mjcf_model.asset.add('mesh', name='upper_arm', file='manipulator/link1.stl', scale=mesh_scale)
        self.mjcf_model.asset.add('mesh', name='forearm', file='manipulator/link2.stl', scale=mesh_scale)

    def _configure_parts(self) -> None:
        self._base = Base(parent=self, name="base", pos=np.zeros(3), euler=[0.0, 0.0, -np.pi / 2])
        self._upper_arm = UpperArm(parent=self._base, name="upper-arm", pos=[0.0, 0.0, 0.07],
                                   euler=[0.0, np.pi / 2, np.pi / 2])
        self._fore_arm = ForeArm(parent=self._upper_arm, name="forearm", pos=[0.3, 0.0, 0.0], euler=[0.0, 0.0, 0.0])
        self._end_effector = EndEffector(parent=self._fore_arm, name="end_effector", pos=[0.31, 0.0, -0.001],
                                         euler=[0.0, 0.0, 0.0])

        self.mjcf_model.contact.add('exclude', name='base_upperarm', body1=self._base.mjcf_body,
                                    body2=self._upper_arm.mjcf_body)
        self.mjcf_model.contact.add('exclude', name='upperarm_forearm', body1=self._upper_arm.mjcf_body,
                                    body2=self._fore_arm.mjcf_body)
        self.mjcf_model.contact.add('exclude', name='forearm_endeffector', body1=self._fore_arm.mjcf_body,
                                    body2=self._end_effector.mjcf_body)

    def after_attachment(self) -> None:
        pass

    def before_substep(self, physics, random_state):
        shoulder_joint = self.mjcf_model.find('joint', 'shoulder')
        elbow_joint = self.mjcf_model.find('joint', 'elbow')

        apply_elastic_actuation(physics=physics,
                                shoulder_joint=shoulder_joint,
                                elbow_joint=elbow_joint,
                                spec=self.specification)

    @property
    def fore_arm(self) -> MJCMorphologyPart:
        return self._fore_arm

    @property
    def upper_arm(self) -> MJCMorphologyPart:
        return self._upper_arm

    @property
    def end_effector(self) -> MJCMorphologyPart:
        return self._end_effector


if __name__ == '__main__':
    morphology = ManipulatorMorphology(specification=ManipulatorMorphologySpecification.default())
    export_with_assets(morphology.mjcf_model, 'robot/robot_xml')
