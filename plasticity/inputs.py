import jax
import jax.numpy as jnp
from typing import Tuple


def generate_input_parameters(
    key: jax.random.PRNGKey, cfg
) -> Tuple[jnp.ndarray, float]:
    """
    Generates input parameters for a neural network model.

    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        cfg: Configuration object containing model settings, including:
            - layer_sizes (list): Sizes of each layer in the network.
            - input_firing_mean (float): Mean firing rate for the input neurons.
            - input_variance (float): Variance for the input neurons.

    Returns:
        Tuple[jnp.ndarray, float]:
            - mus: Array of shape (2, input_dim) representing mean firing rates for two odors.
            - sigma: Standard deviation for the input neurons.
    """
    key, subkey = jax.random.split(key)
    num_odors = 2
    input_dim = cfg.layer_sizes[0]

    # Randomly select half of the input neurons
    firing_idx = jax.random.choice(
        subkey, input_dim, shape=(input_dim // 2,), replace=False
    )
    mask_a = jnp.zeros(input_dim, dtype=bool).at[firing_idx].set(True)
    mask_b = ~mask_a

    # Set means for two different odors
    mus_a = jnp.where(mask_a, cfg.input_firing_mean, 0.0)
    mus_b = jnp.where(mask_b, cfg.input_firing_mean, 0.0)
    mus = jnp.stack([mus_a, mus_b])  # Shape: (2, input_dim)

    # Standard deviation (same for all inputs)
    sigma = jnp.sqrt(cfg.input_variance)

    return mus, sigma


def sample_inputs(
    key: jax.random.PRNGKey, mus: jnp.ndarray, sigma: float, odor: int
) -> jnp.ndarray:
    """
    Samples input data for a given odor using specified mean and variance.

    Args:
        key (jax.random.PRNGKey): Random key for generating random numbers.
        mus (jnp.ndarray): Array of shape (2, input_dim) representing mean firing rates.
        sigma (float): Standard deviation of the input neurons.
        odor (int): Index of the odor to sample inputs for (0 or 1).

    Returns:
        jnp.ndarray: Sampled input data of shape (input_dim,).
    """
    input_dim = mus.shape[1]
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape=(input_dim,)) * sigma
    x = noise + mus[odor]
    return x
