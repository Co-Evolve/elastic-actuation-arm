from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import numpy as np

from erpy.algorithms.map_elites.population import MAPElitesPopulation
from erpy.base.ea import EAConfig
from erpy.base.selector import SelectorConfig, Selector


@dataclass
class MAPElitesSelectorConfig(SelectorConfig):
    @property
    def selector(self) -> Type[MAPElitesSelector]:
        return MAPElitesSelector


class MAPElitesSelector(Selector):
    def __init__(self, config: EAConfig) -> None:
        super(MAPElitesSelector, self).__init__(config=config)

    def select(self, population: MAPElitesPopulation) -> None:
        # Randomly select genomes
        num_to_select = population.population_size - len(population.to_evaluate) - len(population.under_evaluation)
        options = list([descriptor for descriptor, cell in population.archive.items() if
                        cell.genome.genome_id not in population.under_evaluation])

        if num_to_select > 0 and len(options) > 0:
            times_selected = [population.archive_times_selected[cell] for cell in options]
            times_selected = np.argsort(times_selected)

            selected_cells = [options[index] for index in times_selected[:num_to_select]]

            for cell in selected_cells:
                population.archive_times_selected[cell] += 1

            selected_genome_ids = [population.archive[cell].genome.genome_id for cell in selected_cells]
            population.to_reproduce += selected_genome_ids
