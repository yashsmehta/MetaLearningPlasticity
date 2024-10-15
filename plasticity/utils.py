import jax
import re
import jax.numpy as jnp
import numpy as np
from scipy.special import kl_div
from pathlib import Path
import os
import ast
import logging
from pathlib import Path
from typing import Any
from pandas import DataFrame
from filelock import FileLock

# Define color codes for logging
RESET = "\x1b[0m"
COLOR_CODES = {
    logging.DEBUG: "\x1b[37m",     # White
    logging.INFO: "\x1b[32m",      # Green
    logging.WARNING: "\x1b[33m",   # Yellow
    logging.ERROR: "\x1b[31m",     # Red
    logging.CRITICAL: "\x1b[41m",  # Red background
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLOR_CODES.get(record.levelno, RESET)
        message = logging.Formatter.format(self, record)
        return f"{color}{message}{RESET}"

def setup_logging(level=logging.INFO) -> None:
    """Set up logging with colored output."""
    handler = logging.StreamHandler()
    formatter = ColoredFormatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []  # Remove any existing handlers
    root.addHandler(handler)

def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def compute_neg_log_likelihoods(ys, decisions):
    """
    Computes the negative log-likelihoods for a set of predictions and decisions.

    Args:
        ys (array-like): Array of predicted probabilities.
        decisions (array-like): Array of binary decisions (0 or 1).

    Returns:
        float: The mean negative log-likelihood.
    """
    not_ys = jnp.ones_like(ys) - ys
    neg_log_likelihoods = -2 * jnp.log(jnp.where(decisions == 1, ys, not_ys))
    return jnp.mean(neg_log_likelihoods)


def kl_divergence(logits1, logits2):
    """
    Computes the Kullback-Leibler (KL) divergence between two sets of logits.

    Args:
        logits1 (array-like): The first set of logits.
        logits2 (array-like): The second set of logits.

    Returns:
        float: The sum of the KL divergence between the two sets of logits.
    """
    p = jax.nn.softmax(logits1)
    q = jax.nn.softmax(logits2)
    kl_matrix = kl_div(p, q)
    return np.sum(kl_matrix)


def create_nested_list(num_outer, num_inner):
    """
    Creates a nested list with specified dimensions.

    Args:
        num_outer (int): The number of outer lists.
        num_inner (int): The number of inner lists within each outer list.

    Returns:
        list: A nested list where each outer list contains `num_inner` empty lists.
    """
    return [[[] for _ in range(num_inner)] for _ in range(num_outer)]


def truncated_sigmoid(x, epsilon=1e-6):
    """
    Applies a sigmoid function to the input and truncates the output to a specified range.

    Args:
        x (array-like): Input array.
        epsilon (float, optional): Small value to ensure the output is within the range (epsilon, 1 - epsilon). Default is 1e-6.

    Returns:
        array-like: The truncated sigmoid output.
    """
    return jnp.clip(jax.nn.sigmoid(x), epsilon, 1 - epsilon)


def experiment_list_to_tensor(longest_trial_length, nested_list, list_type):
    """
    Converts a nested list of experimental data into a tensor, padding with NaNs or zeros as necessary.

    Args:
        longest_trial_length (int): The length of the longest trial.
        nested_list (list): A nested list containing the experimental data.
        list_type (str): The type of data in the list. Must be one of "decisions", "odors", "xs", or "neural_recordings".

    Returns:
        jnp.ndarray: A tensor representation of the nested list, padded with NaNs or zeros as appropriate.

    Raises:
        Exception: If the list_type is not one of "decisions", "odors", "xs", or "neural_recordings".
    """
    num_blocks = len(nested_list)
    trials_per_block = len(nested_list[0])
    num_trials = num_blocks * trials_per_block

    if list_type == "decisions" or list_type == "odors":
        tensor = np.full((num_trials, longest_trial_length), np.nan)
    elif list_type == "xs" or list_type == "neural_recordings":
        element_dim = len(nested_list[0][0][0])
        tensor = np.full((num_trials, longest_trial_length, element_dim), 0.0)
    else:
        raise Exception("list passed must be odors, decisions or xs")

    for i in range(num_blocks):
        for j in range(trials_per_block):
            trial = nested_list[i][j]
            for k in range(len(trial)):
                tensor[i * trials_per_block + j][k] = trial[k]

    return jnp.array(tensor)


def print_and_log_training_info(cfg, expdata, plasticity_coeff, epoch, loss):
    """
    Logs and prints training information including epoch, loss, and plasticity coefficients.

    Args:
        cfg (object): Configuration object containing model settings and hyperparameters.
        expdata (dict): Dictionary to store experimental data.
        plasticity_coeff (array-like): Array of plasticity coefficients.
        epoch (int): Current epoch number.
        loss (float): Current loss value.

    Returns:
        dict: Updated experimental data dictionary.
    """
    print(f"epoch :{epoch}")
    print(f"loss :{loss}")

    if cfg.plasticity_model == "volterra":
        coeff_mask = np.array(cfg.coeff_mask)
        plasticity_coeff = np.multiply(plasticity_coeff, coeff_mask)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        dict_key = f"A_{i}{j}{k}{l}"
                        expdata.setdefault(dict_key, []).append(
                            plasticity_coeff[i, j, k, l]
                        )

        ind_i, ind_j, ind_k, ind_l = coeff_mask.nonzero()
        top_indices = np.argsort(
            plasticity_coeff[ind_i, ind_j, ind_k, ind_l].flatten()
        )[-5:]
        print("top plasticity coeffs:")
        print("{:<10} {:<20}".format("Term", "Coefficient"))
        for idx in reversed(top_indices):
            term_str = ""
            if ind_i[idx] == 1:
                term_str += "X "
            elif ind_i[idx] == 2:
                term_str += "X² "
            if ind_j[idx] == 1:
                term_str += "Y "
            elif ind_j[idx] == 2:
                term_str += "Y² "
            if ind_k[idx] == 1:
                term_str += "W "
            elif ind_k[idx] == 2:
                term_str += "W² "
            if ind_l[idx] == 1:
                term_str += "R"
            elif ind_l[idx] == 2:
                term_str += "R²"
            coeff = plasticity_coeff[ind_i[idx], ind_j[idx], ind_k[idx], ind_l[idx]]
            print("{:<10} {:<20.5f}".format(term_str, coeff))
        print()
    else:
        print("MLP plasticity coeffs: ", plasticity_coeff)
        expdata.setdefault("mlp_params", []).append(plasticity_coeff)

    expdata.setdefault("epoch", []).append(epoch)
    expdata.setdefault("loss", []).append(loss)

    return expdata


def save_logs(cfg: Any, df: DataFrame) -> Path:
    """
    Saves the logs to a specified directory based on the configuration.

    Args:
        cfg (object): Configuration object containing the model settings and paths.
        df (DataFrame): DataFrame containing the logs to be saved.

    Returns:
        Path: The path where the logs were saved.
    """
    logger = logging.getLogger(__name__)
    logdata_path = Path(cfg.log_dir)

    if cfg.log_expdata:
        subfolder = "expdata" if cfg.use_experimental_data else "simdata"
        logdata_path /= subfolder / cfg.exp_name / cfg.plasticity_model
        logdata_path.mkdir(parents=True, exist_ok=True)

        csv_file = logdata_path / f"exp_{cfg.expid}.csv"
        write_header = not csv_file.exists()
        lock_file = csv_file.with_suffix(".lock")
        lock = FileLock(lock_file)

        try:
            with lock:
                df.to_csv(csv_file, mode="a", header=write_header, index=False)
                logger.info(f"Saved logs to {csv_file}")
        except Exception as e:
            logger.error(f"Failed to save logs to {csv_file}: {e}")
            raise
    else:
        logger.warning("Logging of experimental data is disabled.")

    return logdata_path


def validate_config(cfg: Any) -> Any:
    """
    Validates and processes the configuration object.

    Args:
        cfg: Configuration object containing model settings and paths.

    Returns:
        The validated and processed configuration object.

    Raises:
        ValueError: If any of the configuration validations fail.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Convert layer_sizes from string to list if necessary
    if isinstance(cfg.layer_sizes, str):
        cfg.layer_sizes = ast.literal_eval(cfg.layer_sizes)
        logging.info("Converted layer_sizes from string to list.")

    # Validate reward_ratios length
    if len(cfg.reward_ratios) != cfg.num_blocks:
        raise ValueError("Length of reward_ratios should be equal to num_blocks!")

    # Validate plasticity_model
    if cfg.plasticity_model not in ["volterra", "mlp"]:
        raise ValueError("Only 'volterra' and 'mlp' plasticity models are supported!")

    # Validate generation_model
    if cfg.generation_model not in ["volterra", "mlp"]:
        raise ValueError("Only 'volterra' and 'mlp' generation models are supported!")

    # Validate meta_mlp_layer_sizes
    if not (cfg.meta_mlp_layer_sizes[0] == 4 and cfg.meta_mlp_layer_sizes[-1] == 1):
        raise ValueError("meta_mlp_layer_sizes must start with 4 and end with 1!")

    # Validate output dimension
    if cfg.layer_sizes[-1] != 1:
        raise ValueError("Output dimension (last element of layer_sizes) must be 1!")

    # Validate number of layers
    if len(cfg.layer_sizes) not in [2, 3]:
        raise ValueError("Only 2 or 3 layer networks are supported!")

    # Validate neural_recording_sparsity if fitting neural data
    if "neural" in cfg.fit_data:
        if not (0.0 <= cfg.neural_recording_sparsity <= 1.0):
            raise ValueError("neural_recording_sparsity must be between 0 and 1!")

    # Validate device
    if cfg.device not in ["cpu", "gpu"]:
        raise ValueError("Device must be 'cpu' or 'gpu'!")

    # Validate plasticity_coeff_init for MLP
    if cfg.plasticity_model == "mlp":
        if cfg.plasticity_coeff_init != "random":
            raise ValueError("Only 'random' plasticity_coeff_init is supported for MLP!")

    # Validate fit_data contains 'behavior' or 'neural'
    if not ("behavior" in cfg.fit_data or "neural" in cfg.fit_data):
        raise ValueError("fit_data must contain 'behavior' or 'neural', or both!")

    # If using experimental data, validate related parameters
    if cfg.use_experimental_data:
        num_flies = len(os.listdir(cfg.data_dir))
        if not (0 < cfg.expid <= num_flies):
            raise ValueError(f"Fly experimental data only for fly IDs 1 to {num_flies}!")

        if cfg.num_blocks != 3:
            raise ValueError("All experimental data consists of 3 blocks!")

        # Uncomment the following lines if num_train validation is required
        # if cfg.num_train != 1:
        #     raise ValueError("When fitting per fly, num_train must be 1!")

        if not ("behavior" in cfg.fit_data and "neural" not in cfg.fit_data):
            raise ValueError("Only 'behavior' experimental data is available!")

        # Set certain cfg fields to 'N/A' for experimental data
        cfg.trials_per_block = "N/A"
        cfg.reward_ratios = "N/A"

    # Adjust cfg for 'mlp' plasticity_model
    if cfg.plasticity_model == "mlp":
        cfg.coeff_mask = "N/A"
        cfg.l1_regularization = "N/A"
        cfg.trainable_coeffs = 6 * cfg.meta_mlp_layer_sizes[1] + 1

    # Validate settings for 'volterra' plasticity_model
    if cfg.plasticity_model == "volterra":
        if cfg.log_mlp_plasticity:
            raise ValueError("log_mlp_plasticity must be False for 'volterra' plasticity!")

        if cfg.plasticity_coeff_init not in ["random", "zeros"]:
            raise ValueError(
                "Only 'random' or 'zeros' plasticity_coeff_init supported for 'volterra'!"
            )

    # Adjust cfg fields if not fitting neural data
    if "neural" not in cfg.fit_data:
        cfg.neural_recording_sparsity = "N/A"
        cfg.measurement_noise_scale = "N/A"

    return cfg


def standardize_coeff_init(coeff_init):
    """
    Standardizes the coefficient initialization string by ensuring each variable (X, Y, W, R) is followed by its power.

    Args:
        coeff_init (str): The coefficient initialization string to be standardized.

    Returns:
        str: The standardized coefficient initialization string.
    """
    terms = re.split(r"(?=[+-])", coeff_init)
    formatted_terms = []
    for term in terms:
        var_dict = {"X": 0, "Y": 0, "W": 0, "R": 0}
        number_prefix = re.match(r"[+-]?\d*\.?\d*", term).group(0)
        parts = re.findall(r"([+-]?\d*\.?\d*)([XYWR])(\d*)", term)
        for _, var, power in parts:
            power = int(power) if power else 1
            var_dict[var] = power
        formatted_term = number_prefix + "".join(
            [f"{key}{val}" for key, val in var_dict.items()]
        )
        formatted_terms.append(formatted_term)

    standardized_coeff_init = "".join(formatted_terms)
    return standardized_coeff_init
