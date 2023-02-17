from typing import Union

import numpy as np

from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from erpy.interfaces.mujoco.phenome import MJCMorphologyPart, MJCMorphology
from erpy.utils.colors import rgba_green


class Base(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array,
                 euler: np.array, *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def specification(self) -> ManipulatorMorphologySpecification:
        return super().specification

    def _build(self, *args, **kwargs) -> None:
        I1xx = 0.06335
        I1yy = 0.03294
        I1zz = 0.063300
        I1xy = 0
        I1xz = 0.00650
        I1yz = 0

        self.mjcf_body.add('inertial',
                           pos=[0.0, 0.0, 0.0],
                           mass=self.specification.base_spec.mass.value,
                           fullinertia=[I1xx, I1yy, I1zz, I1xy, I1xz, I1yz])

        self.mjcf_body.add('geom', type='mesh', mesh='base', rgba=rgba_green)
