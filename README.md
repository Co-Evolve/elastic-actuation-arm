# Minimizing torque requirements in robotic manipulation through elastic elements optimization in a physics Engine

[PDF]() | [site](https://sites.google.com/view/elastic-actuation/main)| **Abstract**:  The increasing number of robots and the rising cost of electricity have spurred research into energy-reducing concepts in robotics. One such concept, Elastic Actuation, introduces compliant elements such as springs into the robot structure. This paper presents a comparative analysis between two types of elastic actuation, namely Monoarticular Parallel Elastic Actuation (PEA) and Biarticular Parallel Elastic Actuation (BPEA), and demonstrates an end-to-end pipeline for their optimization. Starting from the real-world system identification of a RRR robotic arm, we calibrate a simulation model in a general-purpose physics engine and employ in-silico evolutionary optimization to co-optimize spring configurations, and trajectories for a pick-and-place task. Finally, we successfully transfer the in-silico optimized elastic elements and trajectory to the real-world prototype. Our results substantiate the ability of elastic actuation to reduce the actuators' torque requirements heavily. Moreover, we highlight the superior performance of the biarticular variant over the monoarticular configuration, with a combination of both proving the most effective. This work provides valuable insights into the torque-reducing use of elastic actuation and demonstrates an actuator-invariant in-silico optimization methodology capable of bridging the sim2real gap.

![overview figure](https://raw.githubusercontent.com/Co-Evolve/elastic-actuation-arm/main/assets/overview.png?token=GHSAT0AAAAAAB5IJXFCEFCDQ6T5KAL7UBCIY77EPTA)


## Usage

Create the anaconda environment:

```shell
conda env create -f environment.yml
```

Run the calibration:

```shell
python -m elastic_actuation_arm.calibration.optimization.main 
```

Run the pick-and-place optimization:

```shell
python -m elastic_actuation_arm.pick_and_place.optimization.main --spring-config {"nea", "pea", "bea", "full"}
```

Metrics are logged and automatically saved on the cloud via [Weights & Biases](https://wandb.ai/site). Genomes are saved locally.

## Citation

```
todo
```
