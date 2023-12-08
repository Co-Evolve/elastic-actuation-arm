from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TYPE_CHECKING, List

from erpy.algorithms.cma_es.population import CMAESPopulation
from erpy.base.genome import ESGenome
from erpy.base.population import Population
from erpy.base.saver import Saver, SaverConfig

if TYPE_CHECKING:
    from erpy.base.ea import EAConfig


@dataclass
class CMAESSaverConfig(SaverConfig):
    @property
    def saver(self) -> Type[CMAESSaver]:
        return CMAESSaver

import io


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        print(module)

        if module == "elastic_actuation_arm.simulation.calibration.optimisation.genome":
            renamed_module = "elastic_actuation_arm.calibration.optimisation.robot.genome"
        elif module == ("elastic_actuation_arm.simulation.pick_and_place.optimisation"
                         ".spring_trajectory_co_optimisatiogenome"):
            renamed_module = "elastic_actuation_arm.pick_and_place.optimisation.robot.genome"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


class CMAESSaver(Saver):
    def __init__(self, config: EAConfig) -> None:
        super().__init__(config)

    def save(self, population: CMAESPopulation) -> None:
        if self.should_save(population.generation):
            # Save the optimizer instance
            path = Path(self.config.save_path) / f"optimizer.pickle"
            with open(path, 'wb') as handle:
                pickle.dump(population.optimizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Save the best genome
            path = Path(self.config.save_path) / f"elite.pickle"
            population.best_genome.save(path)

    def load(self) -> List[ESGenome]:
        elite_path = Path(self.config.save_path) / "elite.pickle"

        with open(elite_path, 'rb') as handle:
            elite = renamed_load(handle)
            # elite = pickle.load(handle)

        return [elite]

    def load_checkpoint(self, checkpoint_path: str, population: Population) -> None:
        raise NotImplementedError

    @property
    def config(self) -> CMAESSaverConfig:
        return self._config
