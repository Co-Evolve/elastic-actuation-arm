import argparse
import logging

import numpy as np
import psutil
import ray

from elastic_actuation_arm.pick_and_place.environment.environment import PickAndPlaceEnvironmentConfig
from elastic_actuation_arm.pick_and_place.optimization.evaluation_callbacks import \
    AdaptEnvironmentConfigCallback, JointVelocityPenaltyCallback, LoadTorqueIntegralSquaredFitnessCallback, \
    LTTrajectoryCallback, TrajectorySaverCallback
from elastic_actuation_arm.pick_and_place.optimization.robot.genome import \
    ManipulatorPickAndPlaceSpringTrajectoryGenomeConfig
from elastic_actuation_arm.pick_and_place.optimization.robot.robot import ManipulatorPickAndPlaceSpringTrajectoryRobot
from erpy.algorithms.cma_es.logger import CMAESLoggerConfig
from erpy.algorithms.cma_es.population import CMAESPopulationConfig
from erpy.algorithms.cma_es.reproducer import CMAESReproducerConfig
from erpy.algorithms.cma_es.saver import CMAESSaverConfig
from erpy.base.ea import EA, EAConfig
from erpy.evaluators.ray.evaluation_actor import make_base_evaluation_actor
from erpy.evaluators.ray.evaluator import RayDistributedEvaluatorConfig
from erpy.selectors.dummy import DummySelectorConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pick And Place - Spring + Trajectory optimization")
    parser.add_argument('--wandb-tags', nargs='+')
    parser.add_argument('--wandb-group', type=str, default='pap-spring-trajectory-co-opt')
    parser.add_argument('--spring-config', default='full', choices=['nea', 'pea', 'bea', 'full',
                                                                    'real_pea', 'real_bea', 'real_full',
                                                                    'real_full_v1'], required=True)
    parser.add_argument("--total-num-cores", type=int, default=psutil.cpu_count())
    parser.add_argument("--output-path", type=str, default='./output')
    parser.add_argument("--from-checkpoint", action='store_true', default=False)
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--cluster", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--analyse-only", action='store_true', default=False)
    parser.add_argument("--analyse-path", type=str)
    args = parser.parse_args()

    seed = np.random.randint(1000000)
    random_state = np.random.RandomState(seed)

    ray.init(log_to_driver=args.debug,
             logging_level=logging.INFO if args.debug else logging.ERROR,
             local_mode=args.debug,
             address="auto" if args.cluster else None)

    genome_config = ManipulatorPickAndPlaceSpringTrajectoryGenomeConfig(
        random_state=random_state,
        spring_config=args.spring_config
    )

    environment_config = PickAndPlaceEnvironmentConfig(seed=seed,
                                                       random_state=random_state,
                                                       go_duration=None,
                                                       ret_duration=None)
    evaluator_config = RayDistributedEvaluatorConfig(
        environment_config=environment_config,
        robot=ManipulatorPickAndPlaceSpringTrajectoryRobot,
        reward_aggregator=np.sum,
        episode_aggregator=np.mean,
        num_workers=args.total_num_cores,
        num_cores_per_worker=1,
        num_eval_episodes=1,
        actor_generator=make_base_evaluation_actor,
        callbacks=[AdaptEnvironmentConfigCallback,
                   LoadTorqueIntegralSquaredFitnessCallback,
                   JointVelocityPenaltyCallback],
        analyze_callbacks=[LoadTorqueIntegralSquaredFitnessCallback, AdaptEnvironmentConfigCallback,
                           LTTrajectoryCallback, TrajectorySaverCallback],
        evaluation_timeout=None,
        hard_episode_reset=False
    )

    logger_config = CMAESLoggerConfig(
        project_name="elastic_actuation_arm",
        group=args.wandb_group,
        tags=args.wandb_tags,
        parameter_labels=genome_config.parameter_labels,
        parameter_rescale_function=genome_config.rescale_parameters,
        update_saver_path=True
    )

    population_config = CMAESPopulationConfig(population_size=64)

    x0 = np.random.rand(genome_config.num_parameters)
    reproducer_config = CMAESReproducerConfig(genome_config=genome_config,
                                              x0=x0,
                                              sigma0=0.15)
    selector_config = DummySelectorConfig()

    saver_config = CMAESSaverConfig(save_freq=1, save_path=args.output_path)

    ea_config = EAConfig(
        num_generations=1000,
        from_checkpoint=args.from_checkpoint,
        checkpoint_path=args.checkpoint_path,
        population_config=population_config,
        evaluator_config=evaluator_config,
        selector_config=selector_config,
        logger_config=logger_config,
        saver_config=saver_config,
        reproducer_config=reproducer_config,
    )

    ea = EA(config=ea_config)
    if not args.analyse_only:
        ea.run()
