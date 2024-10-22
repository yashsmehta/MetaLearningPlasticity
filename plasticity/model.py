import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
import plasticity.data_loader as data_loader
import plasticity.utils as utils
import plasticity.synapse as synapse
import sklearn.metrics
from statistics import mean


def initialize_params(key, cfg, scale=0.01, last_layer_multiplier=5.0):
    """
    Functionality: Initialize parameters for the network.
    Inputs:
        key (int): Seed for the random number generator.
        cfg (object): Configuration object containing the model settings.
        scale (float, optional): Scale for the Gaussian distribution used to initialize the parameters. Default is 0.01.
        last_layer_multiplier (float, optional): Multiplier for the last layer. Default is 5.0.
    Returns: List of tuples of weights and biases for each layer.
    """
    layer_sizes = cfg.layer_sizes

    initial_params = [
        (
            utils.generate_gaussian(key, (m, n), scale),
            # utils.generate_gaussian(key, (n,), scale),
            jnp.zeros((n,)),
        )
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
    ]

    if len(layer_sizes) > 2:
        # for multilayer networks, remove last layer and initialize weights to 1/n
        initial_params.pop()
        hidden_dim = layer_sizes[-2]
        initial_params.append(
            (
                last_layer_multiplier * jnp.ones((hidden_dim, 1)) / hidden_dim,
                jnp.zeros((1,)),
            )
        )

    return initial_params


def network_forward(params, inputs):
    """
    Functionality: Performs a forward pass for the network.
    Inputs:
        params (list): List of tuples (weights, biases) for each layer.
        inputs (array): Input data.
    Returns: Activations for all layers, and logits.
    """
    print("compiling model.network_forward()...")
    activations = [inputs]
    activation = inputs
    for w, b in params[:-1]:
        activation = jnp.tanh(activation @ w + b)
        activations.append(activation)

    final_w, final_b = params[-1]
    logits = activation @ final_w + final_b
    activations.append(logits)
    return activations


@partial(jax.jit, static_argnums=(2,))
def simulate(
    initial_params,
    plasticity_coeffs,
    plasticity_func,
    xs,
    rewards,
    expected_rewards,
    trial_lengths,
):
    """Simulate an experiment with given plasticity coefficients,
       scan over timesteps within a trial, and vmap over the neurons
       (m,n) in plasticity layer (i.e. first layer)
    Inputs:
        initial_params (list): Initial parameters for the network.
        plasticity_coeffs (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        xs (array): Array of inputs.
        rewards (array): Array of rewards.
        expected_rewards (array): Array of expected rewards.
        trial_lengths (array): Array of trial lengths.
    Returns:
        a tensor of activations for the experiment, and the params_trajec,
        i.e. the params at each trial.
        shapes:
            activations: [(num_trials, trial_length)
            weight tensor: (num_trials, input_dim, output_dim)
    """

    print("compiling model.simulate()...")

    def step(carry, stimulus):
        params = carry
        x, reward, expected_reward, trial_length = stimulus
        params, activation = network_step(
            x,
            params,
            plasticity_coeffs,
            plasticity_func,
            reward,
            expected_reward,
            trial_length,
        )
        return params, activation

    final_params, (params_trajec, activations) = jax.lax.scan(
        step, initial_params, (xs, rewards, expected_rewards, trial_lengths)
    )
    return params_trajec, activations


def network_step(
    trial_inputs,
    params,
    plasticity_coeffs,
    plasticity_func,
    reward,
    expected_reward,
    trial_length,
):
    """Performs a forward pass and weight update
        Forward pass is needed to compute logits for the loss function
        for calculating the activations, vmap over all inputs within the trial
    Returns:
        updated params, and stacked: params, logits
    """
    activations = jax.vmap(network_forward, in_axes=(None, 0))(params, trial_inputs)
    # pass only the activations wrt the last odor in trial
    last_odor_activations = [a[trial_length - 1] for a in activations]
    params = update_params(
        params,
        last_odor_activations,
        plasticity_coeffs,
        plasticity_func,
        reward,
        expected_reward,
    )

    return params, (params, activations)


def update_params(
    params, activations, plasticity_coeffs, plasticity_func, reward, expected_reward
):
    """
    Functionality: Updates the parameters of the network, assuming plasticity happens in the first layer only.
    Inputs:
        params (list): List of tuples (weights, biases) for each layer.
        activations (array): Array of activations.
        plasticity_coeffs (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
        reward (float): Reward for the trial.
        expected_reward (float): Expected reward for the trial.
    Returns: Updated parameters.
    """
    print("compiling model.update_params()...")
    input_dim = params[0][0].shape[0]
    lr = 1.0 / input_dim
    # using expected reward or just the reward:
    reward_term = reward - expected_reward
    # reward_term = reward

    delta_params = []

    xs = activations[0]
    ys = activations[1]
    w, b = params[0]
    # vmap over output neurons
    vmap_inputs = jax.vmap(plasticity_func, in_axes=(None, 0, 0, None, None))
    # vmap over input neurons
    vmap_synapses = jax.vmap(vmap_inputs, in_axes=(0, None, 0, None, None))
    dw = vmap_synapses(xs, ys, w, reward_term, plasticity_coeffs)
    # decide whether to update bias or not
    db = jnp.zeros_like(b)
    # db = vmap_inputs(1.0, reward_term, b, plasticity_coeffs)
    assert (
        dw.shape == w.shape and db.shape == b.shape
    ), "dw and w should be of the same shape to prevent broadcasting \
        while adding"
    delta_params.append((dw, db))

    # add the last layer of no plasticity
    if len(params) > len(delta_params):
        delta_params.append((0.0, 0.0))
    params = [
        (w + lr * dw, b + lr * db) for (w, b), (dw, db) in zip(params, delta_params)
    ]

    return params


def evaluate(
    key,
    cfg,
    plasticity_coeff,
    plasticity_func,
):
    """
    Functionality: Evaluates logits, weight trajectory for generation_coeff and plasticity_coeff with new initial params, for a single new experiment.
    Inputs:
        key (int): Seed for the random number generator.
        cfg (object): Configuration object containing the model settings.
        plasticity_coeff (array): Array of plasticity coefficients.
        plasticity_func (function): Plasticity function.
    Returns: R2 score (dict), [weights, activity]; activity is of output of plastic layer. Percent deviance explained (scalar).
    """

    r2_score = {"weights": [], "activity": []}
    percent_deviance = []

    (
        resampled_xs,
        neural_recordings,
        decisions,
        rewards,
        expected_rewards,
    ) = data_loader.load_data(key, cfg, mode="eval")

    for exp_i in decisions:
        key, _ = jax.random.split(key)
        params = initialize_params(key, cfg)
        trial_lengths = data_loader.get_trial_lengths(decisions[exp_i])
        logits_mask = data_loader.get_logits_mask(decisions[exp_i])

        # simulate model with learned plasticity coefficients (plasticity_coeff)
        model_params_trajec, model_activations = simulate(
            params,
            plasticity_coeff,
            plasticity_func,
            resampled_xs[exp_i],
            rewards[exp_i],
            expected_rewards[exp_i],
            trial_lengths,
        )

        # simulate model with zeros plasticity coefficients for null model
        plasticity_coeff_zeros, zero_plasticity_func = synapse.init_plasticity_volterra(
            None, init="zeros"
        )
        _, null_model_activations = simulate(
            params,
            plasticity_coeff_zeros,
            zero_plasticity_func,
            resampled_xs[exp_i],
            rewards[exp_i],
            expected_rewards[exp_i],
            trial_lengths,
        )
        percent_deviance.append(
            evaluate_percent_deviance(
                decisions[exp_i], model_activations, null_model_activations
            )
        )

        if not cfg.use_experimental_data:
            # simulate model with "true" plasticity coefficients (generation_coeff)
            generation_coeff, generation_func = synapse.init_plasticity(
                key, cfg, mode="generation_model"
            )
            params_trajec, activations = simulate(
                params,
                generation_coeff,
                generation_func,
                resampled_xs[exp_i],
                rewards[exp_i],
                expected_rewards[exp_i],
                trial_lengths,
            )

            r2_score_exp = evaluate_r2_score(
                logits_mask,
                params_trajec,
                activations,
                model_params_trajec,
                model_activations,
            )
            r2_score = {
                dict_key: r2_score[dict_key] + r2_score_exp[dict_key]
                for dict_key in r2_score.keys()
            }

    print(f"r2 score: {r2_score}")
    print(f"percent deviance: {percent_deviance}")

    if percent_deviance:
        percent_deviance = np.median(percent_deviance)
        print("median percent deviance: ", percent_deviance)

    if r2_score.get("weights"):
        r2_score["weights"] = np.median(r2_score["weights"])
        print("median r2 weights: ", r2_score["weights"])

    if r2_score.get("activity"):
        r2_score["activity"] = np.median(r2_score["activity"])
        print("median r2 activity: ", r2_score["activity"])
    return r2_score, percent_deviance


def evaluate_r2_score(
    logits_mask, params_trajec, activations, model_params_trajec, model_activations
):
    """
    Functionality: Evaluates the R2 score for weights and activity.
    Inputs:
        logits_mask (array): Mask for the logits.
        params_trajec (array): Array of parameters trajectory.
        activations (array): Array of activations.
        model_params_trajec (array): Array of model parameters trajectory.
        model_activations (array): Array of model activations.
    Returns: A dict of R2 scores for weights and activity in the format of {"weights": [R2 score], "activity": [R2 score]}, i.e. lists of length 1.
    """
    r2_score = {}
    num_trials = logits_mask.shape[0]
    weight_trajec = np.array(params_trajec[0][0]).reshape(num_trials, -1)
    layer_activations = np.squeeze(activations[1])

    model_weight_trajec = np.array(model_params_trajec[0][0]).reshape(num_trials, -1)
    model_layer_activations = np.squeeze(model_activations[1])

    r2_score["weights"] = [sklearn.metrics.r2_score(weight_trajec, model_weight_trajec)]

    if len(params_trajec) == 1:
        # if there is no hidden layer, then the layer activations are the same as the logits
        layer_activations = jax.nn.sigmoid(layer_activations)
        model_layer_activations = jax.nn.sigmoid(model_layer_activations)

    logits_mask = np.where(logits_mask == 0, np.nan, logits_mask)
    layer_activations = layer_activations[~np.isnan(logits_mask)]
    model_layer_activations = model_layer_activations[~np.isnan(logits_mask)]

    r2_score["activity"] = [
        sklearn.metrics.r2_score(layer_activations, model_layer_activations)
    ]
    return r2_score


def evaluate_percent_deviance(decisions, model_activations, null_model_activations):
    """Evaluate logits for plasticity coefficients, calculate
        neg log likelihoods. Then calculate the neg log likelihoods
        where A_ijk = 0 (null model). Then percent deviance explained
        is (D_null - D_model) / D_null
    Returns:
        Percent deviance explained scalar
    """
    ys = jax.nn.sigmoid(jnp.squeeze(model_activations[-1]))
    null_ys = jax.nn.sigmoid(jnp.squeeze(null_model_activations[-1]))
    mask = ~np.isnan(decisions)
    decisions = decisions[mask]
    ys = ys[mask]
    null_ys = null_ys[mask]
    
    model_deviance = utils.compute_neg_log_likelihoods(ys, decisions)
    null_deviance = utils.compute_neg_log_likelihoods(null_ys, decisions)
    percent_deviance = 100 * (null_deviance - model_deviance) / null_deviance
    return percent_deviance.item()
