# Relign

Welcome to the documentation of `Relign`!

# Table Of Contents

1. [Quickstart](quickstart.md)
2. [Reference](reference.md)



Relign is a Python software for simulating the active alignment of optical
systems. More precisely, Relign can be used to analyze the sensor output of
multi-lens systems by changing the relative alignment to the sensor. It also
offers an interface for reinforcement learning algorithms to well-known open
source libraries such as Gymnasium and Stable-Baselines.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

Run the following comments in the project directory:
```sh
pip install .[dev]
```

### Getting started

To familiarize yourself with our the environment for your own algorithms, see
[here](getting_started).

### Reproducibility of research results

To reproduce the results of our paper, have a look at `scripts/train.py`.
A single training can, for instance, be started as follows:


```sh
python3 train.py \
    --env="la" --curriculum --model=PPO \
    --learning-rate=LEARNING_RATE --ent-coef=0.01 \
    --benchmark="b_L2_N000"
```

**Remark:** Currently, this file requires access to WANDB - this will be cleaned up later!



