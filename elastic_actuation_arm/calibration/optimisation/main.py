import argparse
import logging

import numpy as np
import psutil
import ray

from elastic_actuation_arm.calibration.environment.environment import CalibrationEnvironmentConfig
from elastic_actuation_arm.calibration.optimisation.evaluation_callbacks import LoadTorqueNRMSEFitnessCallback, \
    SpringStiffnessPerturbationCallback, TrajectoryPlotCallback, TrajectorySaverCallback, UpdateExperimentIndexCallback
from elastic_actuation_arm.calibration.optimisation.logger import CalibrationLoggerConfig
from elastic_actuation_arm.calibration.optimisation.robot.controller import \
    ManipulatorCalibrationControllerSpecification
from elastic_actuation_arm.calibration.optimisation.robot.genome import ManipulatorCalibrationGenomeConfig
from elastic_actuation_arm.calibration.optimisation.robot.robot import ManipulatorCalibrationRobot
from elastic_actuation_arm.calibration.optimisation.robot.specification import ManipulatorCalibrationSpecification
from elastic_actuation_arm.entities.manipulator.specification import ManipulatorMorphologySpecification
from erpy.algorithms.cma_es.population import CMAESPopulationConfig
from erpy.algorithms.cma_es.reproducer import CMAESReproducerConfig
from erpy.algorithms.cma_es.saver import CMAESSaverConfig
from erpy.base.ea import EA, EAConfig
from erpy.evaluators.ray.evaluation_actor import make_base_evaluation_actor
from erpy.evaluators.ray.evaluator import RayDistributedEvaluatorConfig
from erpy.selectors.dummy import DummySelectorConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Sim2Real calibration"
            )
    parser.add_argument("--wandb-tag", type=str, default="local_testing")
    parser.add_argument("--wandb-group", type=str, default="calibration")
    parser.add_argument("--total-num-cores", type=int, default=psutil.cpu_count())
    parser.add_argument("--cluster", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--output-path", type=str, default='./output')
    parser.add_argument("--analyze-only", action='store_true', default=False)
    parser.add_argument("--analyze-path", type=str)
    args = parser.parse_args()

    seed = np.random.randint(1000000)
    random_state = np.random.RandomState(seed)

    ray.init(
            log_to_driver=args.debug,
            logging_level=logging.INFO if args.debug else logging.ERROR,
            local_mode=args.debug,
            address="auto" if args.cluster else None
            )

    genome_config = ManipulatorCalibrationGenomeConfig(random_state=random_state)
    experiment_ids = [1, 3, 4, 6, 7, 9, 10, 12]

    environment_config = CalibrationEnvironmentConfig(
            seed=seed,
            random_state=random_state,
            real_world_data_path='calibration/environment/real_world_data/version2',
            experiment_ids=experiment_ids
            )

    evaluator_config = RayDistributedEvaluatorConfig(
            environment_config=environment_config,
            robot=ManipulatorCalibrationRobot,
            reward_aggregator=max,
            episode_aggregator=max,
            num_workers=args.total_num_cores,
            num_cores_per_worker=1,
            num_eval_episodes=len(environment_config.experiment_configurations),
            actor_generator=make_base_evaluation_actor,
            callbacks=[UpdateExperimentIndexCallback, SpringStiffnessPerturbationCallback,
                       LoadTorqueNRMSEFitnessCallback],
            analyze_callbacks=[UpdateExperimentIndexCallback, SpringStiffnessPerturbationCallback,
                               LoadTorqueNRMSEFitnessCallback, TrajectorySaverCallback, TrajectoryPlotCallback],
            # TrajectoryPlotCallback],
            hard_episode_reset=True
            )

    logger_config = CalibrationLoggerConfig(
            project_name='elastic_actuation_arm',
            group=args.wandb_group,
            tags=[args.wandb_tag],
            update_saver_path=True,
            parameter_labels=genome_config.parameter_labels,
            error_dimension_labels=environment_config.experiment_configurations[0].measurement_keys,
            parameter_rescale_function=genome_config.rescale_parameters
            )

    population_config = CMAESPopulationConfig(population_size=64)

    x0 = np.random.rand(genome_config.num_parameters)
    reproducer_config = CMAESReproducerConfig(
            genome_config=genome_config, x0=x0, sigma0=0.15
            )
    selector_config = DummySelectorConfig()
    saver_config = CMAESSaverConfig(save_freq=1, save_path=args.output_path)

    ea_config = EAConfig(
            num_generations=int(200),
            population_config=population_config,
            evaluator_config=evaluator_config,
            selector_config=selector_config,
            logger_config=logger_config,
            saver_config=saver_config,
            reproducer_config=reproducer_config
            )

    ea = EA(config=ea_config)

    if not args.analyze_only:
        ea.run()

    # optimised parameters are already stored in the default specification
    specification = ManipulatorCalibrationSpecification.default()
    specification.morphology_specification.end_effector_spec.adhesion.value = False
    genomes, evaluation_results = ea.analyze_specifications(specifications=[specification])
    # genomes, evaluation_results = ea.analyze(args.analyze_path)

    for er in evaluation_results:
        print(f'Specification {er.genome_id}:')
        print(f'\tNRMSE:\t{-er.fitness}')

        nrmse = er.info["logging_nrmse"]
        for name, value in zip(logger_config.error_dimension_labels, nrmse):
            print(f'\t\t{name}: {value}')

        print("\tNRMSE per trajectory")
        nrmse_per_trajectory = er.info["all_nrmses_per_trajectory"]
        shoulder_nrmses = [[], []]
        elbow_nrmses = [[], []]
        for experiment_config, trajectory_nrmse in zip(
                environment_config.experiment_configurations, nrmse_per_trajectory
                ):
            print(f"\t\tTrajectory: {experiment_config.identifier}")
            for name, nrmse in zip(logger_config.error_dimension_labels, trajectory_nrmse):
                if "load_torque" in name:
                    print(f"\t\t\t{name}: {nrmse}")
                    springs = int(experiment_config.spring_stiffness_factor)
                    if "shoulder" in name:
                        shoulder_nrmses[springs].append(nrmse)
                    else:
                        elbow_nrmses[springs].append(nrmse)

        print("\tAveraged LT NRMSEs")
        print(f"\t\tShoulder:")
        print(f"\t\t\tSA:       {np.mean(shoulder_nrmses[0])}")
        print(f"\t\t\tPEA:      {np.mean(shoulder_nrmses[1])}")
        print(f"\t\tElbow:")
        print(f"\t\t\tSA:       {np.mean(elbow_nrmses[0])}")
        print(f"\t\t\tPEA:      {np.mean(elbow_nrmses[1])}")
