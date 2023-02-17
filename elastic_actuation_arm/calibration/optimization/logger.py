from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Optional, List

import numpy as np

from erpy.algorithms.cma_es.logger import CMAESLoggerConfig, CMAESLogger
from erpy.algorithms.cma_es.population import CMAESPopulation
from erpy.base.ea import EAConfig


@dataclass
class CalibrationLoggerConfig(CMAESLoggerConfig):
    error_dimension_labels: Optional[List[str]]

    @property
    def logger(self) -> Type[CalibrationLogger]:
        return CalibrationLogger


class CalibrationLogger(CMAESLogger):
    def __init__(self, config: EAConfig):
        super().__init__(config=config)

    @property
    def config(self) -> CalibrationLoggerConfig:
        return super().config

    def _log_evaluation_result_data(self, population: CMAESPopulation) -> None:
        nrmse = np.array([er.info["logging_nrmse"] for er in population.evaluation_results]).T
        for name, errors in zip(self.config.error_dimension_labels, nrmse):
            self._log_values(name=f"NRMSE/all/{name}", values=errors, step=population.generation)

        # Log NRMSE of current best individual
        nrmse = population.best_er.info["logging_nrmse"]
        for name, value in zip(self.config.error_dimension_labels, nrmse):
            self._log_value(name=f"NRMSE/best/{name}", value=value, step=population.generation)

    def log(self, population: CMAESPopulation) -> None:
        super().log(population)
