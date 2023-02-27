# Minimizing torque requirements in robotic manipulation through Elastic Elements Optimization in a Physics Engine

[PDF]() | [site](https://sites.google.com/view/elastic-actuation/main)| **Abstract**:  todo


todo add overview image


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
