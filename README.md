<img src="./logo.svg" alt="NeurIPS Logo" style="width:30%; height:auto;" align="center"/>

# 🤖🔗🧠 Model Based Inference of Synaptic Plasticity Rules

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?style=for-the-badge&logo=python)](https://docs.python.org/3/whatsnew/3.11.html)
[![JAX](https://img.shields.io/badge/Framework-JAX-important?style=for-the-badge&logo=apache-kafka)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge&logo=open-source-initiative)](https://github.com/countzerozzz/nodepert/edit/master/LICENSE.md)

MetaLearnPlasticity analyzes experimental data from the brain and behavior to discover the rules governing how neurons form and modify their connections (synaptic plasticity). It works by creating mathematical models of these plasticity rules and using gradient descent to find the model parameters that best explain how neural connections change during learning.
#### [🌐 Visit paper website (NeurIPS'24)](https://yashsmehta.com/plasticity-paper-website/)

---
## 🚀 Features

- **Parameterized Plasticity Rules**  
  Models plasticity using Taylor series expansions or multilayer perceptrons for flexibility and interpretability.

- **Gradient Descent Optimization**  
  Optimizes parameters over entire trajectories to match observed neural or behavioral data.

- **Nonlinear Dependencies**  
  Capable of learning rules with intricate, nonlinear dependencies, such as postsynaptic activity and synaptic weights.

- **Simulation Validation**  
  Validates method by recovering known rules (e.g., Oja's rule) and exploring complex hypothetical rules.
---

## 🛠 Installation

This project uses **Poetry** for dependency management. Follow these steps to set up the environment:

1. **Install Poetry**  
   Poetry simplifies dependency management and packaging. Install it with:
   ```bash
   curl -sSL https://install.python-poetry.org | python -
   ```

2. **Clone the Repository**  
   ```bash
   git clone https://github.com/yashsmehta/MetaLearnPlasticity.git
   cd MetaLearnPlasticity
   ```

3. **Install Dependencies**  
   Run the following command to install all required dependencies:
   ```bash
   poetry install
   ```

4. **Activate the Virtual Environment**  
   To activate the project's virtual environment:
   ```bash
   poetry shell
   ```

---

## 📂 Code Structure

- **`plasticity/`**: Main project directory containing core modules:
  - **`run.py`**: Entry point for experiments; configure and start training here.
  - **`synapse.py`**: Handles synaptic plasticity initialization and operations.
  - **`data_loader.py`**: Preprocesses data from experiments or simulations.
  - **`losses.py`**: Defines custom loss functions for optimization.
  - **`model.py`**: Implements the neural network model with parameterized plasticity functions.
  - **`utils.py`**: Contains utility functions for logging and data transformation.
  - **`inputs.py`**: Manages input stimuli for the model.
  - **`trainer.py`**: Encapsulates the training loop, evaluation, and related processes.

- **`pyproject.toml`**: Configuration file for Poetry, listing all project dependencies.

---

## 🧪 Getting Started

1. Set up the environment as described in the **Installation** section.
2. Modify configurations in `run.py` to customize experiments.
3. Run the script:
   ```bash
   python plasticity/run.py
   ```

---

## 📖 Citation

If you use this work, please cite it as follows:

```bibtex
@inproceedings{metalearn-plasticity-2024,
  title={Model-Based Inference of Synaptic Plasticity Rules},
  author={Mehta, Yash and Tyulmankov, Danil and Rajgopalan, Adithya and Turner, Glenn and Fitzgerald, James and Funke, Jan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) 2024},
  year={2024},
  url={https://neurips.cc/},
}
```

---

## 🌟 Contributing

Contributions are welcome! Feel free to submit issues or pull requests. For significant changes, please discuss them via an issue first.

---

## 📜 License

This project is licensed under the [MIT License](https://github.com/countzerozzz/nodepert/edit/master/LICENSE.md).
