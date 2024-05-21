import jax
import jax.numpy as jnp
import numpy as np


def generate_input_parameters(seed, cfg):
    """
    Generates input parameters for a neural network model.

    Args:
        seed (int): Seed for the random number generator to ensure reproducibility.
        cfg (object): Configuration object containing model settings, including:
            - layer_sizes (list): List of integers representing the sizes of each layer in the network.
            - input_firing_mean (float): Mean firing rate for the input neurons.
            - input_variance (float): Variance for the input neurons.

    Returns:
        tuple: A tuple containing:
            - mus (jax.numpy.ndarray): A 2D array of shape (2, input_dim) representing the mean firing rates for two different odors.
            - sigmas (jax.numpy.ndarray): A 3D array of shape (2, input_dim, input_dim) representing the variances for two different odors.
    """
    np.random.seed(seed)
    num_odors = 2
    input_dim = cfg.layer_sizes[0]
    firing_idx = np.random.choice(
        np.arange(input_dim), size=input_dim // 2, replace=False
    )
    mus_a = np.zeros(input_dim)
    mus_a[firing_idx] = cfg.input_firing_mean

    mus_b = cfg.input_firing_mean * np.ones(input_dim)
    mus_b[firing_idx] = 0.0
    mus = np.vstack((mus_a, mus_b))

    diag_mask = np.ma.diag(np.ones(input_dim))

    sigmas = cfg.input_variance * np.ones((num_odors, input_dim, input_dim))

    for i in range(num_odors):
        sigmas[i] = np.multiply(sigmas[i], diag_mask)

    return jnp.array(mus), jnp.array(sigmas)


def sample_inputs(key, mus, sigmas, odor):
    """
    Samples input data for a given odor using specified mean and variance.

    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        mus (jax.numpy.ndarray): 2D array of shape (num_odors, input_dim) representing the mean firing rates for different odors.
        sigmas (jax.numpy.ndarray): 3D array of shape (num_odors, input_dim, input_dim) representing the variances for different odors.
        odor (int): Index of the odor to sample inputs for.

    Returns:
        jax.numpy.ndarray: Sampled input data of shape (input_dim,).
    """
    input_dim = mus.shape[1]
    x = jax.random.normal(key, shape=(input_dim,))
    x = x @ sigmas[odor]
    x = x + mus[odor]
    return x
