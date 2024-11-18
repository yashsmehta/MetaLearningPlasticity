🧠🪰 MetaLearnPlasticity: Inferring Synaptic Plasticity Rules
==========
[![Python 3.10](https://img.shields.io/badge/python-3.11-blue.svg?style=for-the-badge&logo=python)](https://docs.python.org/3/whatsnew/3.11.html)
[![JAX](https://img.shields.io/badge/Framework-JAX-important?style=for-the-badge&logo=Apache-Kafka)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=open-source-initiative)](https://github.com/countzerozzz/nodepert/edit/master/LICENSE.md)


[**Paper Website [NeurIPS'24]**](https://yashsmehta.com/plasticity-paper-website/)


This codebase is designed to infer synaptic plasticity rules from experimental data on neural activity or behavioral trajectories. Our methodology parameterizes the plasticity function to provide theoretical interpretability and facilitate gradient-based optimization.

Features

- Uses Taylor series expansions or multilayer perceptrons to approximate plasticity rules.
- Adjusts parameters via gradient descent over entire trajectories to closely match observed neural activity or behavioral data.
- Can learn intricate rules that induce long nonlinear time-dependencies, such as those incorporating postsynaptic activity and current synaptic weights.
- Validates method through simulations, accurately recovering established rules, like Oja's, as well as more complex hypothetical rules incorporating reward-modulated terms.
- Assesses the resilience of our technique to noise and applies it to behavioral data from Drosophila during a probabilistic reward-learning experiment.

### Installation

This project uses Poetry for dependency management, as indicated by the pyproject.toml file. Here are the detailed installation instructions:

1. Install Poetry: Poetry is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. If you haven't installed Poetry, you can do so by running the following command:
```
curl -sSL https://install.python-poetry.org | python -
```

2. Clone the Repository:
```
git clone https://github.com/yashsmehta/MetaLearnPlasticity.git
```

4. Install Dependencies: Run the following command to install the dependencies of the project:
```
poetry install
```

This command reads the pyproject.toml file from the current directory, resolves the dependencies, and installs them.

5. Activate the Virtual Environment: Poetry creates a virtual environment to isolate the dependencies at a project level. You can activate the created virtual environment by running:
```bash
poetry shell
```

### Code Overview

This project is organized as follows:

- `plasticity/`: This is the main package of the project. It contains the following modules:
- `run.py`: This is the entry point of the project. It sets up the configuration and starts the training process. This is the single file that can be used to run all experiments. Configurations should be changed within this file itself.
- `synapse.py`: This module contains functions related to the initialization and handling of synaptic plasticity.
- `data_loader.py`: This module is responsible for loading and preprocessing the data, either from experimental sources or generated experiments.
- `losses.py`: This module defines the loss functions used in the project.
- `model.py`: This module contains the main model that generates behavioral or neural trajectories with a neural network simulated with a parameterized plasticity function.
- `utils.py`: This module contains utility functions used across the project, such as logging and data transformation utilities.
- `inputs.py`: This module handles the input data (stimulus) for the model.
- `trainer.py`: This module contains the training loop, evaluation, and related functions.
- `pyproject.toml`: This file is used by Poetry for managing project dependencies.

### Citing
If you use our work, please cite:

```bibtex
@inproceedings{metalearn-plasticity-2024,
  title={Model-Based Inference of Synaptic Plasticity Rules},
  author={Mehta, Yash and Tyulmankov, Danil and Rajgopalan, Adithya and Turner, Glenn and Fitzgerald, James and Funke, Jan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) 2024},
  year={2024},
  url={https://neurips.cc/},
  note={To appear}
}
