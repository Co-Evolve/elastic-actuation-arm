from typing import Union

import numpy as np

from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from erpy.interfaces.mujoco.phenome import MJCMorphologyPart, MJCMorphology
from erpy.utils.colors import rgba_green


class ForeArm(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array,
                 euler: np.array, *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def specification(self) -> ManipulatorMorphologySpecification:
        return super().specification

    def _build(self, *args, **kwargs) -> None:
        # Currently in kg * m**2
        I3xx = 0.004809086
        I3yy = 0.042672458
        I3zz = 0.038977764
        I3xy = 0
        I3xz = 0.00024
        I3yz = 0.000002812

        lc3_x = 0.108410  # in meters
        lc3_y = -0.000346
        lc3_z = -0.006379

        self.mjcf_body.add('inertial',
                           pos=[lc3_x, lc3_y, lc3_z],
                           mass=self.specification.fore_arm_spec.mass.value,
                           fullinertia=[I3xx, I3yy, I3zz, I3xy, I3xz, I3yz])

        self.mjcf_body.add('geom', type='mesh', mesh='forearm', rgba=rgba_green)

        self.elbow = self.mjcf_body.add('joint', name='elbow', type='hinge',
                                        axis=[0, 0, -1],
                                        limited=True,
                                        range=self.specification.fore_arm_spec.joint_range.value,
                                        damping=self.specification.fore_arm_spec.joint_damping.value,
                                        armature=self.specification.fore_arm_spec.joint_armature.value,
                                        frictionloss=self.specification.fore_arm_spec.joint_friction_loss.value)

        self.elbow_actuator = self.mjcf_model.actuator.add('position', name="elbow_p_actuator",
                                                           joint=self.elbow,
                                                           ctrllimited=True,
                                                           ctrlrange=self.specification.fore_arm_spec.joint_range.value,
                                                           kp=self.specification.fore_arm_spec.p_actuator_spec.kp.value,
                                                           gear=[
                                                               self.specification.fore_arm_spec.p_actuator_spec.gear.value],
                                                           forcelimited=True,
                                                           forcerange=self.specification.fore_arm_spec.p_actuator_spec.force_limit.value)

        self.mjcf_model.sensor.add('actuatorfrc', name="elbow_actuator_torque",
                                   actuator=self.elbow_actuator)
