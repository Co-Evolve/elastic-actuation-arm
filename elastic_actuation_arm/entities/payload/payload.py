import numpy as np
from dm_control import composer, mjcf

from erpy.utils.colors import rgba_red, rgba_gray


class Payload(composer.Entity):
    def _build(self, mass: float, size: np.ndarray, *args, **kwargs):
        self.mass = mass
        self.size = size

        self._mjcf_model = mjcf.RootElement()
        self._mjcf_body = self.mjcf_model.worldbody.add('body')
        mesh_scale = [0.001, 0.001, 0.001]
        self._mjcf_model.compiler.meshdir = "entities/stl_files"
        self._mjcf_model.asset.add('mesh', name='payload', file='payload/payload.stl', scale=mesh_scale)

        self.mjcf_body.add('geom', name='payload_grab',
                           type='box',
                           size=[0.001, 0.1, 0.1],
                           pos=[-0.001, 0.0, 0.0],
                           mass=0,
                           rgba=rgba_gray,
                           margin=0.0,
                           gap=0.0,
                           condim=1)

        self.mjcf_body.add('geom', name='payload',
                           type='mesh',
                           mesh='payload',
                           mass=mass,
                           rgba=rgba_red,
                           margin=0.01,
                           gap=0.01,
                           condim=1)

    @property
    def mjcf_model(self):
        return self._mjcf_model

    @property
    def mjcf_body(self):
        return self._mjcf_body
