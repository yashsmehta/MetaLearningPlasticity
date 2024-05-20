import jax
import re
import jax.numpy as jnp
from jax import vmap
import numpy as np
from scipy.special import kl_div
from pathlib import Path
import os
import ast
import time
import random


def generate_gaussian(key, shape, scale=0.1):
    """
    returns a random normal tensor of specified shape with zero mean and
    'scale' variance
    """
    assert type(shape) is tuple, "shape passed must be a tuple"
    return scale * jax.random.normal(key, shape)


def compute_neg_log_likelihoods(ys, decisions):
    not_ys = jnp.ones_like(ys) - ys
    neg_log_likelihoods = -2 * jnp.log(jnp.where(decisions == 1, ys, not_ys))
    return jnp.mean(neg_log_likelihoods)


def kl_divergence(logits1, logits2):
    """
    computes the KL divergence between two probability distributions,
    p and q would be logits
    """
    p = jax.nn.softmax(logits1)
    q = jax.nn.softmax(logits2)
    kl_matrix = kl_div(p, q)
    return np.sum(kl_matrix)


def create_nested_list(num_outer, num_inner):
    return [[[] for _ in range(num_inner)] for _ in range(num_outer)]


def truncated_sigmoid(x, epsilon=1e-6):
    return jnp.clip(jax.nn.sigmoid(x), epsilon, 1 - epsilon)


def experiment_list_to_tensor(longest_trial_length, nested_list, list_type):
    # note: trial lengths are not all the same, so we need to pad with nans
    num_blocks = len(nested_list)
    trials_per_block = len(nested_list[0])

    num_trials = num_blocks * trials_per_block

    if list_type == "decisions" or list_type == "odors":
        # individual trial length check for nans and stop when they see one
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
        top_indices = np.argsort(plasticity_coeff[ind_i, ind_j, ind_k, ind_l].flatten())[-5:]
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


def save_logs(cfg, df):
    local_random = random.Random()
    local_random.seed(os.urandom(10))
    sleep_duration = local_random.uniform(1, 5)
    time.sleep(sleep_duration)
    print(f"Slept for {sleep_duration:.2f} seconds.")
    
    logdata_path = Path(cfg.log_dir)
    if cfg.log_expdata:
        if cfg.use_experimental_data:
            logdata_path = (
                logdata_path / "expdata" / cfg.exp_name / cfg.plasticity_model
            )
        else:
            logdata_path = (
                logdata_path / "simdata" / cfg.exp_name / cfg.plasticity_model
            )

        logdata_path.mkdir(parents=True, exist_ok=True)
        csv_file = logdata_path / f"exp_{cfg.expid}.csv"
        write_header = not csv_file.exists()

        # Use a lock file to synchronize access to the CSV file
        lock_file = csv_file.with_suffix(".lock")
        while lock_file.exists():
            print(f"Waiting for lock on {csv_file}...")
            time.sleep(random.uniform(1, 5))

        try:
            lock_file.touch()
            df.to_csv(csv_file, mode="a", header=write_header, index=False)
            print(f"Saved logs to {csv_file}")
        finally:
            lock_file.unlink()

    return logdata_path


def validate_config(cfg):
    """
    asserts that the configuration is valid, and removes any useless keys
    """
    if isinstance(cfg.layer_sizes, str):
        cfg.layer_sizes = ast.literal_eval(cfg.layer_sizes)
        print("passed layer sizes as string, converted to list")
    assert (
        len(cfg.reward_ratios) == cfg.num_blocks
    ), "length of reward ratios should be equal to number of blocks!"
    assert cfg.plasticity_model in [
        "volterra",
        "mlp",
    ], "only volterra, mlp plasticity model supported!"
    assert cfg.generation_model in [
        "volterra",
        "mlp",
    ], "only volterra, mlp generation model supported!"
    assert (
        cfg.meta_mlp_layer_sizes[0] == 4 and cfg.meta_mlp_layer_sizes[-1] == 1
    ), "meta mlp input dim must be 4, and output dim 1!"
    assert cfg.layer_sizes[-1] == 1, "output dim must be 1!"
    assert (
        len(cfg.layer_sizes) == 2 or len(cfg.layer_sizes) == 3
    ), "only 2, 3 layer networks supported!"
    assert (
        cfg.neural_recording_sparsity >= 0.0 and cfg.neural_recording_sparsity <= 1.0
    ), "neural recording sparsity must be between 0 and 1!"
    assert cfg.device in ["cpu", "gpu"], "device must be cpu or gpu!"
    if cfg.plasticity_model == "mlp":
        assert cfg.plasticity_coeff_init in [
            "random"
        ], "only random plasticity coeff init for MLP supported!"
    assert (
        "behavior" in cfg.fit_data or "neural" in cfg.fit_data
    ), "fit data must contain either behavior or neural, or both!"
    if cfg.use_experimental_data:
        num_flies = len(os.listdir(cfg.data_dir))
        assert (
            cfg.flyid > 0 and cfg.flyid <= num_flies
        ), f"Fly experimental data only for flyids 1-{num_flies}! "
        assert cfg.num_blocks == 3, "all Adi's data gathering consists of 3 blocks!"
        assert (
            "behavior" in cfg.fit_data and "neural" not in cfg.fit_data
        ), "only behavior experimental data available!"
        cfg["trials_per_block"] = "N/A"
        cfg["reward_ratios"] = "N/A"

    if cfg["plasticity_model"] == "mlp":
        cfg["coeff_mask"] = "N/A"
        cfg["l1_regularization"] = "N/A"
        cfg["trainable_coeffs"] = 6 * (cfg["meta_mlp_layer_sizes"][1]) + 1
    if cfg["plasticity_model"] == "volterra":
        assert cfg["log_mlp_plasticity"] == False, "log_mlp_plasticity must be False for volterra plasticity!"
        assert cfg.plasticity_coeff_init in [
            "random",
            "zeros",
        ], "only random or zeros plasticity coeff init for volterra supported!"
    if "neural" not in cfg["fit_data"]:
        cfg["neural_recording_sparsity"] = "N/A"
        cfg["measurement_noise_scale"] = "N/A"
    return cfg


def standardize_coeff_init(coeff_init):
    terms = re.split(r'(?=[+-])', coeff_init)
    formatted_terms = []
    for term in terms:
        var_dict = {'X': 0, 'Y': 0, 'W': 0, 'R': 0}
        number_prefix = re.match(r'[+-]?\d*\.?\d*', term).group(0)
        parts = re.findall(r'([+-]?\d*\.?\d*)([XYWR])(\d*)', term)
        for _, var, power in parts:
            power = int(power) if power else 1
            var_dict[var] = power
        formatted_term = number_prefix + ''.join([f"{key}{val}" for key, val in var_dict.items()])
        formatted_terms.append(formatted_term)

    standardized_coeff_init = ''.join(formatted_terms)
    return standardized_coeff_init
