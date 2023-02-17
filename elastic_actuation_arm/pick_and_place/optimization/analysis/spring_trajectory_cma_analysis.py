import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import psutil
import ray
from elastic_actuation_arm.pick_and_place.optimization.spring.common.evaluation_callbacks import \
    LTTrajectoryCallback, TrajectorySaverCallback, ParametersPhenomeDescriptorCallback, \
    LoadTorqueIntegralSquaredFitnessCallback, LoadTorqueRMSLoggerCallback, TrajectoryPlotCallback
from tqdm import tqdm

from elastic_actuation_arm.pick_and_place.environment.environment import PickAndPlaceEnvironmentConfig
from elastic_actuation_arm.pick_and_place.optimization.evaluation_callbacks import \
    AdaptEnvironmentConfigCallback, JointVelocityPenaltyCallback
from elastic_actuation_arm.pick_and_place.optimization.robot.genome import \
    ManipulatorPickAndPlaceSpringTrajectoryGenomeConfig
from elastic_actuation_arm.pick_and_place.optimization.robot import \
    ManipulatorPickAndPlaceSpringTrajectoryRobot
from erpy.algorithms.cma_es.logger import CMAESLoggerConfig
from erpy.algorithms.cma_es.population import CMAESPopulationConfig
from erpy.algorithms.cma_es.reproducer import CMAESReproducerConfig
from erpy.algorithms.cma_es.saver import CMAESSaverConfig
from erpy.base.ea import EA, EAConfig
from erpy.evaluators.evaluation_callbacks.video import VideoCallback
from erpy.evaluators.ray.evaluation_actor import make_base_evaluation_actor
from erpy.evaluators.ray.evaluator import RayDistributedEvaluatorConfig
from erpy.selectors.dummy import DummySelectorConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pick And Place - Spring + Trajectory optimization")
    parser.add_argument('--wandb-tags', nargs='+')
    parser.add_argument('--wandb-group', type=str, default='pap-spring-trajectory-co-opt')
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
        spring_config=None
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
        callbacks=[ParametersPhenomeDescriptorCallback,
                   AdaptEnvironmentConfigCallback,
                   LoadTorqueIntegralSquaredFitnessCallback,
                   LoadTorqueRMSLoggerCallback],
        analyze_callbacks=[LoadTorqueIntegralSquaredFitnessCallback, AdaptEnvironmentConfigCallback,
                           LTTrajectoryCallback, TrajectoryPlotCallback, TrajectorySaverCallback,
                           LoadTorqueRMSLoggerCallback, JointVelocityPenaltyCallback,
                           VideoCallback
                           ],
        evaluation_timeout=None,
        hard_episode_reset=False
    )

    logger_config = CMAESLoggerConfig(
        project_name="elastic_actuation_arm",
        group=args.wandb_group,
        tags=args.wandb_tags,
        parameter_labels=[],
        parameter_rescale_function=genome_config.rescale_parameters,
        update_saver_path=True
    )

    population_config = CMAESPopulationConfig(population_size=64)

    x0 = np.random.rand(3)
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

    spring_configs = ["nea", "pea", "bea", "full"]
    all_genomes = []
    all_evaluation_results = []
    all_run_paths = []

    base_path = Path(args.analyse_path)
    for spring_config in spring_configs:
        print(f"ANALYSING {spring_config}")
        genome_config.spring_config = spring_config

        spring_config_path = base_path / spring_config

        run_paths = glob.glob(str(spring_config_path / "*-*-*"))

        genomes = []
        evaluation_results = []
        for run_path in tqdm(run_paths):
            # Do elite analysis per run
            ea_config.saver_config.save_path = run_path
            ea.evaluator._build_pool()
            genome = ea.saver.load()[0]
            genomes.append(genome)

            evaluation_result = ea.analyze_genomes([genome])[0]
            evaluation_results.append(evaluation_result)

        all_evaluation_results.append(evaluation_results)
        all_genomes.append(genomes)
        all_run_paths.append(run_paths)


    def print_best_mean_std(name, values, best_index):
        print(f"\t{name}\t{values[best_index]:.4f}/{np.mean(values):.4f}/{np.std(values):.4f}")


    for genomes, evaluation_results, spring_config, run_paths in zip(all_genomes, all_evaluation_results,
                                                                     spring_configs, all_run_paths):
        genome_config.spring_config = spring_config

        # Get the best genome
        fitnesses = [evaluation_result.fitness for evaluation_result in evaluation_results]
        best_index = np.argmax(fitnesses)

        best_genome = evaluation_results[best_index]
        best_er = evaluation_results[best_index]

        rescaled_parameters = np.array([genome_config.rescale_parameters(genome.parameters) for genome in genomes]).T
        best_parameters = rescaled_parameters[best_index]

        print(f"Spring config: {spring_config}")
        print(f"\tVelocity invalidity: {evaluation_results[best_index].info['JointVelocityLimitCallback']}")
        print(f"\tBest genome run path: {run_paths[best_index]}")
        print(f"\tBest genome index:    {best_genome.genome_id}")
        print_best_mean_std("Load torque integral", np.abs(fitnesses), best_index)
        print(f"\t\tAll load torques: {list(np.abs(fitnesses))}")

        for name, parameters in zip(genome_config.parameter_labels, rescaled_parameters):
            if 'equilibrium_angle' in name or 'q0' in name:
                parameters = parameters * 180 / np.pi
            if "trajectory_point" in name:
                continue
            print_best_mean_std(name, parameters, best_index)

        print()
        print()
