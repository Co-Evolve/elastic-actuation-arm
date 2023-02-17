from typing import Union

import numpy as np

from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from erpy.interfaces.mujoco.phenome import MJCMorphologyPart, MJCMorphology


class EndEffector(MJCMorphologyPart):
    def __init__(self, parent: Union[MJCMorphology, MJCMorphologyPart], name: str, pos: np.array,
                 euler: np.array, *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def specification(self) -> ManipulatorMorphologySpecification:
        return super().specification

    def _build(self, *args, **kwargs) -> None:
        self._end_effector = self.mjcf_body.add('geom', name='end_effector', type='box',
                                                pos=[0.0, 0.0, 0.0],
                                                size=[0.05, 0.05, 0.01],
                                                euler=[0, np.pi / 2, 0],
                                                rgba=[0, 0, 0, 0],
                                                mass=0)
        self.mjcf_body.add('site', name='end_effector_site', type='box',
                           pos=[0.0, 0.0, 0.0],
                           size=[0.05, 0.05, 0.01],
                           euler=[0, np.pi / 2, 0],
                           rgba=[0, 0, 0, 0])

        if self.specification.end_effector_spec.adhesion.value:
            self.mjcf_model.actuator.add('adhesion', name='end_effector_suction_actuator',
                                         body=self.mjcf_body, gain=2000, ctrlrange=[0.0, 1.0])
