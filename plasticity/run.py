from omegaconf import OmegaConf
import numpy as np
import logging

import plasticity.trainer as trainer
from plasticity.utils import setup_platform, validate_config, setup_logging

def create_default_config():
    """Create and return the default configuration dictionary."""
    # coeff_mask = np.zeros((3, 3, 3, 3))
    # coeff_mask[0:2, 0, 0, 0:2] = 1
    coeff_mask = np.ones((3, 3, 3, 3))

    return {
        "exp_type": "train",
        "use_experimental_data": False,
        "num_train": 20,
        "num_eval": 5,
        "expid": 5,
        "num_epochs": 250,
        "trials_per_block": 80,
        "num_blocks": 3,
        "device": "cpu",
        "reward_ratios": ((0.2, 0.8), (0.9, 0.1), (0.2, 0.8)),
        "log_expdata": True,
        "log_mlp_plasticity": False,
        "log_interval": 25,
        "fit_data": "behavior",
        "neural_recording_sparsity": 1.0,
        "measurement_noise_scale": 0.0,
        "layer_sizes": "[2, 10, 1]",
        "input_firing_mean": 0.75,
        "input_variance": 0.015,
        "regularization_type": "l1",
        "regularization_scale": 1e-2,
        "generation_coeff_init": "X1Y0W0R1",
        "generation_model": "volterra",
        "plasticity_coeff_init": "random",
        "plasticity_model": "volterra",
        "meta_mlp_layer_sizes": [4, 10, 1],
        "moving_avg_window": 10,
        "data_dir": "../data/",
        "log_dir": "logs/",
        "learning_rate": 1e-3,
        "trainable_coeffs": int(np.sum(coeff_mask)),
        "coeff_mask": coeff_mask.tolist(),
        "exp_name": "base",
        "reward_term": "expected_reward",
    }

def main():
    """Main entry point for running the training or evaluation process."""
    
    setup_logging(level=logging.INFO)
    cfg_dict = create_default_config()
    cfg = OmegaConf.create(cfg_dict)
    
    # Merge with command-line arguments
    config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, config)
    
    # Validate the configuration
    cfg = validate_config(cfg)
    setup_platform(cfg.device)
    
    # Log important configuration parameters
    if cfg.use_experimental_data:
        logging.info(f"Using experimental data from {cfg.data_dir}")
    else:
        logging.info(f"Using simulated data")
        logging.info(f"Ground truth plasticity rule: {cfg.generation_coeff_init}")

    logging.info(f"Fitting on: {cfg.fit_data} trajectories")
    logging.info(f"Plasticity model: {cfg.plasticity_model}")
    logging.info(f"Neural network layer sizes: {cfg.layer_sizes}")
    logging.info(f"Trainable coefficients: {cfg.trainable_coeffs}")
    
    trainer.train(cfg)

if __name__ == "__main__":
    main()


"""
Main entry point for running the training or evaluation process.

This script initializes the configuration for the process, sets up the necessary parameters,
and starts the training or evaluation using the specified trainer and evaluator.

Configuration Parameters:
- exp_type (str): Type of experiment ("train" or "eval").
- num_train (int): Number of trajectories in the training set, each simulated with different initial weights.
- num_eval (int): Number of evaluation trajectories.
- num_epochs (int): Number of training epochs.
- trials_per_block (int): Total length of trajectory is number of blocks * trials per block.
- num_blocks (int): Each block can have different reward probabilities/ratios for the odors.
- device (str): Device to run the training on (e.g., "cpu").
- reward_ratios (tuple): A:B reward probabilities corresponding to each block.
- log_expdata (bool): Flag to save the training data.
- log_mlp_plasticity (bool): Flag to save the trajectory of the plasticity MLP weights.
- log_interval (int): Log training data every x epochs.
- use_experimental_data (bool): Use simulated or experimental data.
- expid (int): Experiment ID saved for parallel runs on cluster.
- fit_data (str): Type of data to fit on ("neural" or "behavior").
- neural_recording_sparsity (float): Sparsity of neural recordings (1.0 means all neurons are recorded).
- measurement_noise_scale (float): Scale of Gaussian noise added to neural recordings.
- layer_sizes (str): Network layer sizes [input_dim, hidden_dim, output_dim].
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
- trainable_coeffs (int): Number of trainable coefficients.
- coeff_mask (list): Mask for the coefficients.
- exp_name (str): Name under which logs are stored.
- reward_term (str): Reward term to use ("reward" or "expected_reward").
"""
