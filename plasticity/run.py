import plasticity.trainer as trainer
from omegaconf import OmegaConf
import numpy as np


if __name__ == "__main__":
    # coeff_mask is used to vary the trainable parameters in the taylor series.
    # so coeff_mask[:,:,0] = 1, would pass the gradients through only term in the taylor series without weight dependence (since W^0)
    coeff_mask = np.zeros((3, 3, 3, 3))
    coeff_mask[:, :, :, :] = 1

    cfg_dict = {
        "num_train": 1,  # number of trajectories in the training set. each simulated one with different initial weights.
        "num_eval": 10,
        "num_epochs": 250,
        "trials_per_block": 80,  # total length of trajectory is number of blocks * trails per block
        "num_blocks": 3,  # each block can have different reward probabilities/ratios for the odors
        "reward_ratios": (
            (0.2, 0.8),
            (0.9, 0.1),
            (0.2, 0.8),
        ),  # A:B reward probabilities for corresponding to each block
        "log_expdata": True,  # flag to save the training data
        "log_interval": 25,  # log training data every x epochs
        "use_experimental_data": False,  # use simulated or experimental data
        "flyid": 1,  # flyid saved for parallel runs on cluster
        "fit_data": "behavior",  # code searches for words: "neural", "behavior", corresponding to fitting on neural activity recordings or binary behavioral choices
        "neural_recording_sparsity": 1.0,  # sparsity of 1. means all neurons are recorded
        "measurement_noise_scale": 0.0,  # scale of gaussian noise added to neural recordings would be measurement_noise * firing_rate
        "layer_sizes": "[2, 10, 1]",  # network [input_dim, hidden_dim, output_dim], only 2, 3 layer network supported, since we're modeling plasticity in a single layer
        "input_firing_mean": 0.75,  # stimulus is encoded as firing neurons. mean value of firing input neuron
        "input_variance": 0.05,  # variance of input encoding of stimulus
        "l1_regularization": 5e-2,  # L1 regularizer on the taylor series params to enforce sparse solutions
        "generation_coeff_init": "X1Y0R1W0",  # simulated data underlying rule, integers are the corresponding powers, so X1R1W0 would be XR.
        "generation_model": "volterra",  # "volterra" (this just means taylor, it's called volterra in the code, to be consistent with prior literature) or "mlp"
        "plasticity_coeff_init": "random",  # initializations for the parameters of the plasticity rule, "random" or "zeros"
        "plasticity_model": "mlp",  # "volterra" or "mlp (what is the functional family of the plasticity model)
        "meta_mlp_layer_sizes": [
            4,
            10,
            1,
        ],  # [4, hidden_dim, 1] (if functional family is MLP, then define parameters)
        "moving_avg_window": 10,  # define the window for calculating expected reward, E[R]
        "data_dir": "../../data/",  # directory to load experimental data
        "log_dir": "logs/",  # directory to save experimental data
        "trainable_coeffs": int(np.sum(coeff_mask)),
        "coeff_mask": coeff_mask.tolist(),
        "exp_name": "trial",  # logs are stored under a directory created under this name
        "reward_term": "expected_reward",  # reward or expected_reward. Note, this is a dummy config, one needs to manually change the reward_term variable, in model.update_params()
    }

    cfg = OmegaConf.create(cfg_dict)

    config = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, config)

    trainer.train(cfg)
