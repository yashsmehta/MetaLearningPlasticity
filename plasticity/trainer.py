import time
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import jax
from jax.random import split
import optax

import plasticity.synapse as synapse
import plasticity.data_loader as data_loader
import plasticity.losses as losses
import plasticity.model as model
import plasticity.utils as utils


def initialize_parameters(
    cfg: Dict[str, Any], key: jax.random.PRNGKey
) -> Tuple[Any, Any, Any, jax.random.PRNGKey]:
    """Initialize model parameters and plasticity coefficients."""
    key, subkey = split(key)
    params = model.initialize_params(key, cfg)
    plasticity_coeff, plasticity_func = synapse.init_plasticity(
        subkey, cfg, mode="plasticity_model"
    )
    return params, plasticity_coeff, plasticity_func, key

def training_loop(
    cfg: Dict[str, Any],
    params: Any,
    plasticity_coeff: Any,
    plasticity_func: Any,
    data: Tuple[Any, Any, Any, Any, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """Run the training loop."""
    loss_value_and_grad = jax.value_and_grad(losses.loss, argnums=2)
    optimizer = optax.adam(learning_rate=cfg['learning_rate'])
    opt_state = optimizer.init(plasticity_coeff)
    expdata: Dict[str, Any] = {}
    noise_key = jax.random.PRNGKey(10 * cfg['expid'])
    resampled_xs, neural_recordings, decisions, rewards, expected_rewards = data

    for epoch in range(cfg['num_epochs'] + 1):
        for exp_i in decisions:
            noise_key, _ = split(noise_key)
            loss, meta_grads = loss_value_and_grad(
                noise_key,
                params,
                plasticity_coeff,
                plasticity_func,
                resampled_xs[exp_i],
                rewards[exp_i],
                expected_rewards[exp_i],
                neural_recordings[exp_i],
                decisions[exp_i],
                cfg,
            )
            updates, opt_state = optimizer.update(
                meta_grads, opt_state, plasticity_coeff
            )
            plasticity_coeff = optax.apply_updates(plasticity_coeff, updates)

        if epoch % cfg['log_interval'] == 0:
            expdata = utils.print_and_log_training_info(
                cfg, expdata, plasticity_coeff, epoch, loss
            )
    return plasticity_coeff, expdata

def evaluate_model(
    cfg: Dict[str, Any],
    plasticity_coeff: Any,
    plasticity_func: Any,
    key: jax.random.PRNGKey,
    expdata: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate the trained model."""
    if cfg['num_eval'] > 0:
        logging.info("Evaluating model...")
        r2_score, percent_deviance = model.evaluate(
            key,
            cfg,
            plasticity_coeff,
            plasticity_func,
        )
        expdata["percent_deviance"] = percent_deviance
        if not cfg['use_experimental_data']:
            expdata["r2_weights"] = r2_score["weights"]
            expdata["r2_activity"] = r2_score["activity"]
    return expdata

def save_results(cfg: Dict[str, Any], expdata: Dict[str, Any], train_time: float) -> str:
    """Save training logs and parameters."""
    df = pd.DataFrame.from_dict(expdata)
    df["train_time"] = train_time

    # Add configuration parameters to DataFrame
    for cfg_key, cfg_value in cfg.items():
        if isinstance(cfg_value, (float, int, str)):
            df[cfg_key] = cfg_value
    df["layer_sizes"] = str(cfg['layer_sizes'])

    logging.info(df.tail(5))
    logdata_path = utils.save_logs(cfg, df)
    return logdata_path


def train(cfg: Dict[str, Any]) -> None:
    """
    Train a neural network model based on the provided configuration.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing model settings and hyperparameters.
    """
    key = jax.random.PRNGKey(cfg['expid'])
    data = data_loader.load_data(key, cfg)
    params, plasticity_coeff, plasticity_func, key = initialize_parameters(cfg, key)

    start_time = time.time()
    plasticity_coeff, expdata = training_loop(
        cfg, params, plasticity_coeff, plasticity_func, data
    )
    train_time = round(time.time() - start_time, 2)
    logging.info(f"Training time: {train_time}s")

    # Remove 'mlp_params' from expdata if present
    mlp_params = expdata.pop("mlp_params", None)

    key, _ = split(key)
    expdata = evaluate_model(cfg, plasticity_coeff, plasticity_func, key, expdata)
    logging.info(f"Percent deviance explained: {expdata['percent_deviance']}")
    logdata_path = save_results(cfg, expdata, train_time)

    # Save MLP parameters if required
    if cfg['plasticity_model'] == "mlp" and cfg['log_mlp_plasticity'] and mlp_params:
        with open(Path(logdata_path) / f"mlp_params_{cfg['expid']}.pkl", "wb") as f:
            pickle.dump(mlp_params, f)
