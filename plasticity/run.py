"""
Main entry point for running the training process.

This script initializes the configuration for the training process, sets up the necessary parameters,
and starts the training using the specified trainer.

Configuration Parameters:
- coeff_mask (numpy.ndarray): Mask to vary the trainable parameters in the Taylor series.
- num_train (int): Number of trajectories in the training set, each simulated with different initial weights.
- num_eval (int): Number of evaluation trajectories.
- num_epochs (int): Number of training epochs.
- trials_per_block (int): Total length of trajectory is number of blocks * trials per block.
- num_blocks (int): Each block can have different reward probabilities/ratios for the odors.
- reward_ratios (list of tuples): A:B reward probabilities corresponding to each block.
- log_expdata (bool): Flag to save the training data.
- log_interval (int): Log training data every x epochs.
- use_experimental_data (bool): Use simulated or experimental data.
- expid (int): Fly ID saved for parallel runs on cluster.
- device (str): Device to run the training on (e.g., "cpu").
- fit_data (str): Type of data to fit on ("neural" or "behavior").
- neural_recording_sparsity (float): Sparsity of neural recordings (1.0 means all neurons are recorded).
- measurement_noise_scale (float): Scale of Gaussian noise added to neural recordings.
- layer_sizes (list): Network layer sizes [input_dim, hidden_dim, output_dim].
- input_firing_mean (float): Mean value of firing input neuron.
- input_variance (float): Variance of input encoding of stimulus.
- l1_regularization (float): L1 regularizer on the Taylor series parameters to enforce sparse solutions.
- generation_coeff_init (str): Initialization string for the generation coefficients.
- generation_model (str): Model type for generation ("volterra" or "mlp").
- plasticity_coeff_init (str): Initialization method for the plasticity coefficients ("random" or "zeros").
- plasticity_model (str): Model type for plasticity ("volterra" or "mlp").
- meta_mlp_layer_sizes (list): Layer sizes for the MLP if the functional family is MLP.
- moving_avg_window (int): Window size for calculating expected reward, E[R].
- data_dir (str): Directory to load experimental data.
- log_dir (str): Directory to save experimental data.
- learning_rate (float): Learning rate for the optimizer.
- trainable_coeffs (int): Number of trainable coefficients.
- coeff_mask (list): Mask for the coefficients.
- exp_name (str): Name under which logs are stored.
- reward_term (str): Reward term to use ("reward" or "expected_reward").

The configuration is created from a dictionary and can be overridden by command-line arguments.
Finally, the training process is started using the specified trainer.
"""

import numpy as np
from omegaconf import OmegaConf
import plasticity.trainer as trainer


def main() -> None:
    """Run the training process with the specified configuration."""
    coeff_mask = np.ones((3, 3, 3, 3))

    cfg_dict = {
        "num_train": 5,
        "num_eval": 10,
        "num_epochs": 25,
        "trials_per_block": 80,
        "num_blocks": 3,
        "reward_ratios": [
            (0.2, 0.8),
            (0.9, 0.1),
            (0.2, 0.8),
        ],
        "log_expdata": False,
        "log_interval": 25,
        "use_experimental_data": False,
        "expid": 1,
        "device": "cpu",
        "fit_data": "neural",
        "neural_recording_sparsity": 1.0,
        "measurement_noise_scale": 0.0,
        "layer_sizes": [2, 10, 1],
        "input_firing_mean": 0.75,
        "input_variance": 0.05,
        "l1_regularization": 5e-2,
        "generation_coeff_init": "X1Y0R1W0",
        "generation_model": "volterra",
        "plasticity_coeff_init": "random",
        "plasticity_model": "volterra",
        "meta_mlp_layer_sizes": [4, 10, 1],
        "moving_avg_window": 10,
        "data_dir": "../../data/",
        "log_dir": "logs/",
        "learning_rate": 1e-3,
        "trainable_coeffs": int(np.sum(coeff_mask)),
        "coeff_mask": coeff_mask.tolist(),
        "exp_name": "trial",
        "reward_term": "expected_reward",
    }

    cfg = OmegaConf.create(cfg_dict)

    # Override configuration with command-line arguments if provided
    cli_config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_config)

    trainer.train(cfg)


if __name__ == "__main__":
    main()
