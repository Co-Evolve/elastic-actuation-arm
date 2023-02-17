from typing import Union

import numpy as np

from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from erpy.interfaces.mujoco.phenome import MJCMorphologyPart, MJCMorphology
from erpy.utils.colors import rgba_green


class UpperArm(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array,
                 euler: np.array, *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def specification(self) -> ManipulatorMorphologySpecification:
        return super().specification

    def _build(self, *args, **kwargs) -> None:
        # Currently in kg * m**2
        I2xx = 0.012784922
        I2yy = 0.040284487
        I2zz = 0.030942239
        I2xy = 0.000096092
        I2xz = 0.003278807
        I2yz = 0

        lc2_x = 0.080193  # in meters
        lc2_y = -0.001080
        lc2_z = -0.018886

        self.mjcf_body.add('inertial',
                           pos=[lc2_x, lc2_y, lc2_z],
                           mass=self.specification.upper_arm_spec.mass.value,
                           fullinertia=[I2xx, I2yy, I2zz, I2xy, I2xz, I2yz])

        self.mjcf_body.add('geom', type='mesh', mesh='upper_arm', rgba=rgba_green)

        self.shoulder = self.mjcf_body.add('joint', name='shoulder', type='hinge',
                                           axis=[0, 0, 1],
                                           limited=True,
                                           range=self.specification.upper_arm_spec.joint_range.value,
                                           damping=self.specification.upper_arm_spec.joint_damping.value,
                                           armature=self.specification.upper_arm_spec.joint_armature.value,
                                           frictionloss=self.specification.upper_arm_spec.joint_friction_loss.value)

        self.shoulder_actuator = self.mjcf_model.actuator.add('position',
                                                              name="shoulder_p_actuator",
                                                              joint=self.shoulder,
                                                              ctrllimited=True,
                                                              ctrlrange=self.specification.upper_arm_spec.joint_range.value,
                                                              kp=self.specification.upper_arm_spec.p_actuator_spec.kp.value,
                                                              gear=[
                                                                  self.specification.upper_arm_spec.p_actuator_spec.gear.value],
                                                              forcelimited=True,
                                                              forcerange=self.specification.upper_arm_spec.p_actuator_spec.force_limit.value)

        self.mjcf_model.sensor.add('actuatorfrc', name="shoulder_actuator_torque",
                                   actuator=self.shoulder_actuator)
