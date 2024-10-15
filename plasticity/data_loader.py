import numpy as np
import jax
from jax.random import bernoulli, split
from jax.nn import sigmoid
import jax.numpy as jnp
import collections
import scipy.io as sio
import os
from functools import partial
import logging
from typing import Any, Dict, Tuple

import plasticity.model as model
import plasticity.inputs as inputs
import plasticity.synapse as synapse
from plasticity.utils import experiment_list_to_tensor
from plasticity.utils import create_nested_list

logger = logging.getLogger(__name__)

def load_data(key, cfg, mode="train"):
    """
    Functionality: Load data for training or evaluation.
    Inputs:
        key (int): Seed for the random number generator.
        cfg (object): Configuration object containing the model settings.
        mode (str, optional): Mode of operation ("train" or "eval"). Default is "train".
    Returns: Depending on the configuration, either experimental data or generated experiments data.
    """
    assert mode in ["train", "eval"]
    if cfg.use_experimental_data:
        return load_fly_expdata(key, cfg, mode)

    else:
        generation_coeff, generation_func = synapse.init_plasticity(
            key, cfg, mode="generation_model"
        )
        return generate_experiments_data(
            key, cfg, generation_coeff, generation_func, mode
        )


def generate_experiments_data(key, cfg, plasticity_coeff, plasticity_func, mode):
    """
    Functionality: Simulate all fly experiments with given plasticity coefficients.
    Inputs:
        key (int): Seed for the random number generator.
        cfg (object): Configuration object containing the model settings.
        plasticity_coeff (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        mode (str): Mode of operation ("train" or "eval").
    Returns: 5 dictionaries corresponding with experiment number (int) as key, with tensors as values.
    """
    if mode == "train":
        num_experiments = cfg.num_train
    else:
        num_experiments = cfg.num_eval
    xs, odors, neural_recordings, decisions, rewards, expected_rewards = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    print("generating experiments data...")

    for exp_i in range(num_experiments):
        seed = (cfg.expid + 1) * (exp_i + 1)
        odor_mus, odor_sigmas = inputs.generate_input_parameters(seed, cfg)
        exp_i = str(exp_i)
        key, subkey = split(key)
        params = model.initialize_params(key, cfg)
        # print("prob_output:")
        (
            exp_xs,
            exp_odors,
            exp_neural_recordings,
            exp_decisions,
            exp_rewards,
            exp_expected_rewards,
        ) = generate_experiment(
            subkey,
            cfg,
            params,
            plasticity_coeff,
            plasticity_func,
            odor_mus,
            odor_sigmas,
        )

        trial_lengths = [
            [len(exp_decisions[j][i]) for i in range(cfg.trials_per_block)]
            for j in range(cfg.num_blocks)
        ]
        max_trial_length = np.max(np.array(trial_lengths))
        print("Exp " + exp_i + f", longest trial length: {max_trial_length}")

        xs[exp_i] = experiment_list_to_tensor(max_trial_length, exp_xs, list_type="xs")
        odors[exp_i] = experiment_list_to_tensor(
            max_trial_length, exp_odors, list_type="odors"
        )
        neural_recordings[exp_i] = experiment_list_to_tensor(
            max_trial_length, exp_neural_recordings, list_type="neural_recordings"
        )
        decisions[exp_i] = experiment_list_to_tensor(
            max_trial_length, exp_decisions, list_type="decisions"
        )
        rewards[exp_i] = np.array(exp_rewards, dtype=float).flatten()
        expected_rewards[exp_i] = np.array(exp_expected_rewards, dtype=float).flatten()
        # print("odors: ", odors[exp_i])
        # print("rewards: ", rewards[exp_i])

    return xs, neural_recordings, decisions, rewards, expected_rewards


def generate_experiment(
    key,
    cfg,
    params,
    plasticity_coeffs,
    plasticity_func,
    odor_mus,
    odor_sigmas,
):
    """
    Functionality: Simulate a single fly experiment with given plasticity coefficients.
    Inputs:
        key (int): Seed for the random number generator.
        cfg (object): Configuration object containing the model settings.
        params (list): List of tuples (weights, biases) for each layer.
        plasticity_coeffs (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        odor_mus (array): Array of odor means.
        odor_sigmas (array): Array of odor standard deviations.
    Returns: A nested list (num_blocks x trials_per_block) of lists of different lengths corresponding to the number of timesteps in each trial.
    """

    r_history = collections.deque(
        cfg.moving_avg_window * [0], maxlen=cfg.moving_avg_window
    )
    rewards_in_arena = np.zeros(
        2,
    )

    xs, odors, neural_recordings, decisions, rewards, expected_rewards = (
        create_nested_list(cfg.num_blocks, cfg.trials_per_block) for _ in range(6)
    )

    for block in range(cfg.num_blocks):
        r_ratio = cfg.reward_ratios[block]
        for trial in range(cfg.trials_per_block):
            key, _ = split(key)
            sampled_rewards = bernoulli(key, np.array(r_ratio))
            rewards_in_arena = np.logical_or(sampled_rewards, rewards_in_arena)
            key, _ = split(key)

            trial_data, params, rewards_in_arena, r_history = generate_trial(
                key,
                params,
                plasticity_coeffs,
                plasticity_func,
                rewards_in_arena,
                r_history,
                odor_mus,
                odor_sigmas,
            )
            (
                xs[block][trial],
                odors[block][trial],
                neural_recordings[block][trial],
                decisions[block][trial],
                rewards[block][trial],
                expected_rewards[block][trial],
            ) = trial_data

    return xs, odors, neural_recordings, decisions, rewards, expected_rewards


def generate_trial(
    key,
    params,
    plasticity_coeffs,
    plasticity_func,
    rewards_in_arena,
    r_history,
    odor_mus,
    odor_sigmas,
):
    """
    Functionality: Simulate a single fly trial, which ends when the fly accepts odor.
    Inputs:
        key (int): Seed for the random number generator.
        params (list): List of tuples (weights, biases) for each layer.
        plasticity_coeffs (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        rewards_in_arena (array): Array of rewards in the arena.
        r_history (deque): History of rewards.
        odor_mus (array): Array of odor means.
        odor_sigmas (array): Array of odor standard deviations.
    Returns: A tuple containing lists of xs, odors, decisions (sampled outputs), rewards, and expected_rewards for the trial.
    """

    input_xs, trial_odors, neural_recordings, decisions = [], [], [], []

    expected_reward = np.mean(r_history)

    while True:
        key, _ = split(key)
        odor = int(bernoulli(key, 0.5))
        trial_odors.append(odor)
        key, subkey = split(key)
        x = inputs.sample_inputs(key, odor_mus, odor_sigmas, odor)
        resampled_x = inputs.sample_inputs(subkey, odor_mus, odor_sigmas, odor)
        jit_network_forward = jax.jit(model.network_forward)
        activations = jit_network_forward(params, x)
        prob_output = sigmoid(activations[-1])

        key, subkey = split(key)
        sampled_output = float(bernoulli(subkey, prob_output))

        input_xs.append(resampled_x)
        # always recording the output neuron
        neural_recordings.append(prob_output)

        decisions.append(sampled_output)

        if sampled_output == 1:
            # print(prob_output)
            reward = rewards_in_arena[odor]
            r_history.appendleft(reward)
            rewards_in_arena[odor] = 0
            jit_update_params = partial(jax.jit, static_argnums=(3,))(
                model.update_params
            )
            params = jit_update_params(
                params,
                activations,
                plasticity_coeffs,
                plasticity_func,
                reward,
                expected_reward,
            )
            break

    return (
        (input_xs, trial_odors, neural_recordings, decisions, reward, expected_reward),
        params,
        rewards_in_arena,
        r_history,
    )


def expected_reward_for_exp_data(R, moving_avg_window):
    """
    Functionality: Calculate expected rewards for experimental data.
    Inputs:
        R (array): Array of rewards.
        moving_avg_window (int): Size of the moving average window.
    Returns: Array of expected rewards.
    """
    r_history = collections.deque(moving_avg_window * [0], maxlen=moving_avg_window)
    expected_rewards = []
    for r in R:
        expected_rewards.append(np.mean(r_history))
        r_history.appendleft(r)
    return np.array(expected_rewards)


def load_fly_expdata(
    key: Any, cfg: Any, mode: str
) -> Tuple[Dict[int, np.ndarray], Dict[int, Any], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Load experimental data for training or evaluation.

    Args:
        key (Any): Random number generator key or seed.
        cfg (Any): Configuration object containing the model settings.
        mode (str): Mode of operation ("train" or "eval").

    Returns:
        Tuple[Dict[int, np.ndarray], ...]: Dictionaries for xs, neural_recordings, decisions, rewards, expected_rewards.
    """
    logger.info(f"Loading {mode} experimental data...")

    xs: Dict[int, np.ndarray] = {}
    neural_recordings: Dict[int, Any] = {}
    decisions: Dict[int, np.ndarray] = {}
    rewards: Dict[int, np.ndarray] = {}
    expected_rewards: Dict[int, np.ndarray] = {}

    max_exp_id = 18  # Total number of fly data files
    input_dim = cfg.layer_sizes[0]
    num_sampling = cfg.num_train if mode == "train" else cfg.num_eval

    for sample_idx in range(num_sampling):
        key, subkey = split(key)
        file_number = ((cfg.expid + sample_idx - 1) % max_exp_id) + 1
        file_name = f"Fly{file_number}.mat"
        file_path = os.path.join(cfg.data_dir, file_name)

        odor_mus, odor_sigmas = inputs.generate_input_parameters(key, cfg)
        logger.info(f"File {file_name}, loading sample id {sample_idx}")

        try:
            data = sio.loadmat(file_path)
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            continue

        odor_ids = data.get("X")
        Y = np.squeeze(data.get("Y"))
        R = np.squeeze(data.get("R"))

        if odor_ids is None or Y is None or R is None:
            logger.error(f"Data missing in file: {file_path}")
            continue

        odors = np.where(odor_ids == 1)[1]
        num_trials = int(np.sum(Y))
        if num_trials != R.shape[0]:
            logger.error("Y and R should have the same number of trials")
            continue

        # Compute the starting index for each trial
        indices = np.insert(np.cumsum(Y), 0, 0)[:-1].astype(int)

        # Initialize lists to collect decisions and inputs per trial
        exp_decisions = [[] for _ in range(num_trials)]
        exp_xs = [[] for _ in range(num_trials)]

        for idx, decision, odor in zip(indices, Y, odors):
            exp_decisions[idx].append(decision)
            x = inputs.sample_inputs(subkey, odor_mus, odor_sigmas, odor)
            exp_xs[idx].append(x)

        trial_lengths = [len(trial) for trial in exp_decisions]
        max_trial_length = max(trial_lengths)
        logger.info(f"Max trial length for sample {sample_idx}: {max_trial_length}")

        # Prepare tensors for decisions and inputs
        decisions_tensor = np.full((num_trials, max_trial_length), np.nan)
        xs_tensor = np.zeros((num_trials, max_trial_length, input_dim))

        for i, (decisions_list, xs_list) in enumerate(zip(exp_decisions, exp_xs)):
            length = len(decisions_list)
            decisions_tensor[i, :length] = decisions_list
            xs_tensor[i, :length, :] = xs_list

        # Store data in dictionaries
        decisions[str(sample_idx)] = decisions_tensor
        xs[str(sample_idx)] = xs_tensor
        rewards[str(sample_idx)] = R
        expected_rewards[str(sample_idx)] = expected_reward_for_exp_data(R, cfg.moving_avg_window)
        neural_recordings[str(sample_idx)] = None

    return xs, neural_recordings, decisions, rewards, expected_rewards


def get_trial_lengths(decisions):
    """
    Functionality: Get the lengths of trials.
    Inputs:
        decisions (array): Array of decisions.
    Returns: Array of trial lengths.
    """
    trial_lengths = jnp.sum(jnp.logical_not(jnp.isnan(decisions)), axis=1).astype(int)
    return trial_lengths


def get_logits_mask(decisions):
    """
    Functionality: Get a mask for the logits.
    Inputs:
        decisions (array): Array of decisions.
    Returns: Array representing the mask for the logits.
    """
    logits_mask = jnp.logical_not(jnp.isnan(decisions)).astype(int)
    return logits_mask
