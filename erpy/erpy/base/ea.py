from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

from erpy.base.evaluator import EvaluatorConfig, Evaluator, EvaluationResult
from erpy.base.genome import Genome, DummyGenome
from erpy.base.logger import LoggerConfig, Logger
from erpy.base.population import PopulationConfig, Population
from erpy.base.reproducer import ReproducerConfig, Reproducer
from erpy.base.saver import SaverConfig, Saver
from erpy.base.selector import SelectorConfig, Selector
from erpy.base.specification import RobotSpecification


@dataclass
class EAConfig:
    population_config: PopulationConfig
    evaluator_config: EvaluatorConfig
    selector_config: SelectorConfig
    reproducer_config: ReproducerConfig
    logger_config: LoggerConfig
    saver_config: SaverConfig

    cli_args: Optional[Dict[str, Any]] = None
    from_checkpoint: bool = False
    checkpoint_path: str = None
    num_generations: Optional[int] = None
    num_evaluations: Optional[int] = None

    @property
    def population(self) -> Population:
        return self.population_config.population(self)

    @property
    def evaluator(self) -> Evaluator:
        return self.evaluator_config.evaluator(self)

    @property
    def selector(self) -> Selector:
        return self.selector_config.selector(self)

    @property
    def reproducer(self) -> Reproducer:
        return self.reproducer_config.reproducer(self)

    @property
    def logger(self) -> Logger:
        return self.logger_config.logger(self)

    @property
    def saver(self) -> Saver:
        return self.saver_config.saver(self)


class EA:
    def __init__(self, config: EAConfig):
        self._config = config

        self._population = None
        self._evaluator = None
        self._selector = None
        self._reproducer = None
        self._logger = None
        self._saver = None

    @property
    def config(self) -> EAConfig:
        return self._config

    @property
    def population(self) -> Population:
        if self._population is None:
            self._population = self.config.population
        return self._population

    @property
    def evaluator(self) -> Evaluator:
        if self._evaluator is None:
            self._evaluator = self.config.evaluator
        return self._evaluator

    @property
    def selector(self) -> Selector:
        if self._selector is None:
            self._selector = self.config.selector
        return self._selector

    @property
    def reproducer(self) -> Reproducer:
        if self._reproducer is None:
            self._reproducer = self.config.reproducer
        return self._reproducer

    @property
    def logger(self) -> Logger:
        if self._logger is None:
            self._logger = self.config.logger
        return self._logger

    @property
    def saver(self) -> Saver:
        if self._saver is None:
            self._saver = self.config.saver
        return self._saver

    def is_done(self) -> bool:
        is_done = False
        if self.config.num_generations is not None and self.population.generation >= self.config.num_generations:
            is_done = True
        if self.config.num_evaluations is not None and self.population.num_evaluations >= self.config.num_evaluations:
            is_done = True
        return is_done

    def run(self) -> None:
        if self.config.from_checkpoint:
            self.saver.load_checkpoint(checkpoint_path=self.config.checkpoint_path,
                                       population=self.population)
            self.reproducer.initialise_from_checkpoint(population=self.population)
            self.selector.select(population=self.population)
        else:
            self.reproducer.initialise_population(self.population)

        while not self.is_done():
            self.reproducer.reproduce(population=self.population)
            self.population.before_evaluation()
            self.evaluator.evaluate(population=self.population)
            self.population.after_evaluation()
            self.logger.log(population=self.population)
            self.saver.save(population=self.population)
            self.selector.select(population=self.population)
            self.population.generation += 1

        self._evaluator = None

    def load_genomes(self, path: Optional[str] = None) -> List[Genome]:
        if path is not None:
            self.config.saver_config.save_path = path

        return self.saver.load()

    def analyze(self, path: Optional[str] = None) -> Tuple[List[Genome], List[EvaluationResult]]:
        genomes = self.load_genomes(path)

        return genomes, self.analyze_genomes(genomes)

    def analyze_specifications(self, specifications: List[RobotSpecification]) -> Tuple[
        List[Genome], List[EvaluationResult]]:
        genomes = [DummyGenome(genome_id=i, specification=specification) for i, specification in
                   enumerate(specifications)]
        return genomes, self.analyze_genomes(genomes=genomes)

    def analyze_genomes(self, genomes: List[Genome]) -> List[EvaluationResult]:
        # Reset population by recreating it
        self._population = None

        for genome in genomes:
            self.population.genomes[genome.genome_id] = genome
            self.population.to_evaluate.append(genome.genome_id)

        self.evaluator.evaluate(self.population, analyze=True)

        # Sort evaluation results according to genome order
        er_genome_ids = [er.genome_id for er in self.population.evaluation_results]
        er_indices = [er_genome_ids.index(genome.genome_id) for genome in genomes]

        return [self.population.evaluation_results[i] for i in er_indices]
