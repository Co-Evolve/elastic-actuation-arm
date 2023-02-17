from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Optional, List

import numpy as np

from elastic_actuation_arm.pick_and_place.optimization.robot.specification import \
    ManipulatorPickAndPlaceSpringTrajectorySpecification
from erpy.base.genome import ESGenome, Genome, ESGenomeConfig
from erpy.base.parameters import ContinuousParameter, Parameter, DiscreteParameter
from erpy.utils.math import renormalize


@dataclass
class ManipulatorPickAndPlaceSpringTrajectoryGenomeConfig(ESGenomeConfig):
    spring_config: str = 'pea'

    @property
    def genome(self) -> Type[ManipulatorPickAndPlaceSpringTrajectoryGenome]:
        return ManipulatorPickAndPlaceSpringTrajectoryGenome

    def extract_morphology_parameters(self, specification: ManipulatorPickAndPlaceSpringTrajectorySpecification) -> \
            np.ndarray[ContinuousParameter]:
        morph_spec = specification.morphology_specification
        morphology_parameters = []
        if self.spring_config == "pea" or self.spring_config == "full":
            morphology_parameters += [morph_spec.upper_arm_spec.spring_spec.stiffness,
                                      morph_spec.upper_arm_spec.spring_spec.equilibrium_angle,

                                      morph_spec.fore_arm_spec.spring_spec.stiffness,
                                      morph_spec.fore_arm_spec.spring_spec.equilibrium_angle,
                                      ]

        if self.spring_config == "bea" or self.spring_config == "full":
            morphology_parameters += [morph_spec.biarticular_spring_spec.stiffness,
                                      morph_spec.biarticular_spring_spec.q0,
                                      morph_spec.biarticular_spring_spec.r1,
                                      morph_spec.biarticular_spring_spec.r2,
                                      morph_spec.biarticular_spring_spec.r3,
                                      ]

        if self.spring_config == "real_pea" or self.spring_config == "real_full":
            morphology_parameters += [morph_spec.upper_arm_spec.spring_spec.equilibrium_angle,
                                      morph_spec.fore_arm_spec.spring_spec.equilibrium_angle]

        if self.spring_config == "real_bea" or self.spring_config == "real_full":
            morphology_parameters += [morph_spec.biarticular_spring_spec.q0]

        if self.spring_config == "real_full_v1":
            morphology_parameters = [morph_spec.biarticular_spring_spec.q0]

        return morphology_parameters

    def extract_controller_parameters(self, specification: ManipulatorPickAndPlaceSpringTrajectorySpecification) -> \
            np.ndarray[ContinuousParameter]:
        control_spec = specification.controller_specification
        controller_parameters = np.concatenate((control_spec.go_intermediate_q[:, 1:].flatten(),
                                                control_spec.ret_intermediate_q[:, 1:].flatten(),
                                                [control_spec.go_duration,
                                                 control_spec.ret_duration]))
        return controller_parameters

    def extract_parameters(self, specification: ManipulatorPickAndPlaceSpringTrajectorySpecification) -> np.ndarray[
        Parameter]:
        morphology_parameters = self.extract_morphology_parameters(specification)
        controller_parameters = self.extract_controller_parameters(specification)
        parameters = np.concatenate((morphology_parameters, controller_parameters))
        return parameters

    def base_specification(self) -> ManipulatorPickAndPlaceSpringTrajectorySpecification:
        if self.spring_config == "nea":
            return ManipulatorPickAndPlaceSpringTrajectorySpecification.nea_default()
        if self.spring_config == "pea":
            return ManipulatorPickAndPlaceSpringTrajectorySpecification.pea_default()
        if self.spring_config == "bea":
            return ManipulatorPickAndPlaceSpringTrajectorySpecification.bea_default()
        if self.spring_config == "full":
            return ManipulatorPickAndPlaceSpringTrajectorySpecification.default()
        if self.spring_config == "real_pea":
            return ManipulatorPickAndPlaceSpringTrajectorySpecification.real_pea_default()
        if self.spring_config == "real_bea":
            return ManipulatorPickAndPlaceSpringTrajectorySpecification.real_bea_default()
        if self.spring_config == "real_full":
            return ManipulatorPickAndPlaceSpringTrajectorySpecification.real_full_default()
        if self.spring_config == "real_full_v1":
            return ManipulatorPickAndPlaceSpringTrajectorySpecification.real_full_v1_default()
        assert False, f"Unrecognised spring config requested: {self.spring_config}"

    @property
    def parameter_labels(self) -> List[str]:
        morphology_labels, controller_labels = [], []
        if self.spring_config == 'pea' or self.spring_config == 'full':
            morphology_labels += ["PEA_shoulder_stiffness", "PEA_shoulder_equilibrium_angle", "PEA_elbow_stiffness",
                                  "PEA_elbow_equilibrium_angle"]
        if self.spring_config == 'bea' or self.spring_config == 'full':
            morphology_labels += ["BEA_stiffness", "BEA_q0", "BEA_r1", "BEA_r2", "BEA_r3"]

        if self.spring_config == "real_pea" or self.spring_config == "real_full":
            morphology_labels += ["PEA_shoulder_equilibrium_angle",
                                  "PEA_elbow_equilibrium_angle"]

        if self.spring_config == "real_bea" or self.spring_config == "real_full":
            morphology_labels += ["BEA_q0"]

        if self.spring_config == "real_full_v1":
            morphology_labels = ["BEA_q0"]

        num_points = len(self.extract_controller_parameters(self.base_specification())) - 2
        controller_labels += [f"trajectory_point_{i}" for i in range(num_points)] + ["go_duration", "ret_duration"]

        return morphology_labels + controller_labels

    def rescale_parameters(self, parameters: np.ndarray) -> np.ndarray:
        spec = self.base_specification()
        default_params = self.extract_parameters(spec)

        rescaled_parameters = []
        for param, value in zip(default_params, parameters):
            if isinstance(param, ContinuousParameter):
                value = renormalize(value, [0, 1], [param.low, param.high])
            rescaled_parameters.append(value)

        return rescaled_parameters


class ManipulatorPickAndPlaceSpringTrajectoryGenome(ESGenome):
    def __init__(self, parameters: np.ndarray,
                 config: ManipulatorPickAndPlaceSpringTrajectoryGenomeConfig,
                 genome_id: int, parent_genome_id: Optional[int] = None):
        super().__init__(parameters=parameters, config=config, genome_id=genome_id, parent_genome_id=parent_genome_id)

    @property
    def config(self) -> ManipulatorPickAndPlaceSpringTrajectoryGenomeConfig:
        return super().config

    @staticmethod
    def generate(config: ManipulatorPickAndPlaceSpringTrajectoryGenomeConfig,
                 genome_id: int, *args, **kwargs) -> ManipulatorPickAndPlaceSpringTrajectoryGenome:
        base_spec = config.base_specification()
        default_parameters = config.extract_parameters(base_spec)

        parameters = []
        for parameter in default_parameters:
            if isinstance(parameter, ContinuousParameter):
                parameters.append(config.random_state.rand())
            elif isinstance(parameter, DiscreteParameter):
                parameters.append(config.random_state.choice(parameter.options))
            else:
                raise AssertionError(f"Undefined parameter: {parameter}")

        return ManipulatorPickAndPlaceSpringTrajectoryGenome(parameters=parameters, config=config, genome_id=genome_id)

    @property
    def specification(self) -> ManipulatorPickAndPlaceSpringTrajectorySpecification:
        return super().specification

    def mutate(self, child_genome_id: int, *args, **kwargs) -> ESGenome:
        pass

    def cross_over(self, partner_genome: Genome, child_genome_id: int) -> ESGenome:
        pass
