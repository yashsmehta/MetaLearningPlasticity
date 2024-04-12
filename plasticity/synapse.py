from plasticity.utils import generate_gaussian
import jax
import jax.numpy as jnp
import numpy as np
import re


def volterra_synapse_tensor(x, y, w, r):
    """
    Functionality: Computes the Volterra synapse tensor for given inputs.
    Inputs: x, y, w, r (floats): Inputs to the Volterra synapse tensor.
    Returns: A 3x3x3x3 tensor representing the Volterra synapse tensor.
    """
    synapse_tensor = jnp.array(
        [
            [
                [
                    [x**i * y**j * w**k * r**l for i in range(3)]
                    for j in range(3)
                ]
                for k in range(3)
            ]
            for l in range(3)
        ]
    )
    return synapse_tensor


def volterra_plasticity_function(x, y, w, r, volterra_coefficients):
    """
    Functionality: Computes the Volterra plasticity function for given inputs and coefficients.
    Inputs: x, y, w, r (floats): Inputs to the Volterra plasticity function.
            volterra_coefficients (array): Coefficients for the Volterra plasticity function.
    Returns: The result of the Volterra plasticity function.
    """
    synapse_tensor = volterra_synapse_tensor(x, y, w, r)
    dw = jnp.sum(jnp.multiply(volterra_coefficients, synapse_tensor))
    return dw


def mlp_forward_pass(mlp_params, inputs):
    """
    Functionality: Performs a forward pass through a multi-layer perceptron (MLP).
    Inputs: mlp_params (list): List of tuples (weights, biases) for each layer.
            inputs (array): Input data.
    Returns: The logits output of the MLP.
    """
    activation = inputs
    for w, b in mlp_params[:-1]:  # for all but the last layer
        activation = jax.nn.leaky_relu(jnp.dot(activation, w) + b)
    final_w, final_b = mlp_params[-1]  # for the last layer
    logits = jnp.dot(activation, final_w) + final_b
    output = jnp.tanh(logits)
    return jnp.squeeze(output)


def mlp_plasticity_function(x, y, w, r, mlp_params):
    """
    Functionality: Computes the MLP plasticity function for given inputs and MLP parameters.
    Inputs: x, y, z (floats): Inputs to the MLP plasticity function.
            mlp_params (list): MLP parameters.
    Returns: The result of the MLP plasticity function.
    """
    inputs = jnp.array([x, y, w, r])
    dw = mlp_forward_pass(mlp_params, inputs)
    return dw


def init_zeros():
    return np.zeros((3, 3, 3, 3))


def init_random(key):
    assert key is not None, "For random initialization, a random key has to be given"
    return generate_gaussian(key, (3, 3, 3, 3), scale=1e-5)


def split_init_string(s):
    """
    Functionality: Splits an initialization string into a list of matches.
    Inputs: s (str): Initialization string.
    Returns: A list of matches.
    """
    return [
        match.replace(" ", "")
        for match in re.findall(r"(-?\s*[A-Za-z0-9.]+[A-Za-z][0-9]*)", s)
    ]


def extract_numbers(s):
    """
    Functionality: Extracts numbers from string initialization: X1R0W1
    for the plasticity coefficients
    Inputs: s (str): String to extract numbers from.
    Returns: A tuple of extracted numbers.
    """
    x = int(re.search("X(\d+)", s).group(1))
    y = int(re.search("Y(\d+)", s).group(1))
    w = int(re.search("W(\d+)", s).group(1))
    r = int(re.search("R(\d+)", s).group(1))
    multiplier_match = re.search("^(-?\d+\.?\d*)", s)
    multiplier = float(multiplier_match.group(1)) if multiplier_match else 1.0
    assert x < 3 and y < 3 and w < 3 and r < 3, "X, Y, W, R must be between 0 and 2"
    return x, y, w, r, multiplier


def init_generation_volterra(init):
    """
    Functionality: Initializes the parameters for the Volterra generation model.
    Inputs: init (str): Initialization string.
    Returns: A tuple containing the initialized parameters and the Volterra plasticity function.
    """
    parameters = np.zeros((3, 3, 3, 3))
    inits = split_init_string(init)
    for init in inits:
        x, y, w, r, multiplier = extract_numbers(init)
        parameters[x][y][w][r] = multiplier

    return jnp.array(parameters), volterra_plasticity_function


def init_plasticity_volterra(key, init):
    init_functions = {
        "zeros": init_zeros,
        "random": lambda: init_random(key),
    }

    parameters = init_functions[init]()
    return jnp.array(parameters), volterra_plasticity_function


def init_plasticity_mlp(key, layer_sizes, scale=0.01):

    mlp_params = [
        (
            generate_gaussian(key, (m, n), scale),
            generate_gaussian(key, (n,), scale),
        )
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
    return mlp_params, mlp_plasticity_function


def init_plasticity(key, cfg, mode):
    """
    Functionality: Initializes the parameters for a given plasticity model.
    Inputs: key (int): Seed for the random number generator.
            cfg (object): Configuration object containing the model settings.
            mode (str): Mode of operation ("generation" or "plasticity").
    Returns: A tuple containing the initialized parameters and the corresponding plasticity function.
    """
    if "generation" in mode:
        if cfg.generation_model == "volterra":
            return init_generation_volterra(init=cfg.generation_coeff_init)
        elif cfg.generation_model == "mlp":
            return init_plasticity_mlp(key, cfg.meta_mlp_layer_sizes)
    elif "plasticity" in mode:
        if cfg.plasticity_model == "volterra":
            return init_plasticity_volterra(key, init=cfg.plasticity_coeff_init)
        elif cfg.plasticity_model == "mlp":
            return init_plasticity_mlp(key, cfg.meta_mlp_layer_sizes)

    raise RuntimeError(
        f"mode needs to be either generation or plasticity, and plasticity_model needs to be either volterra or mlp"
    )
