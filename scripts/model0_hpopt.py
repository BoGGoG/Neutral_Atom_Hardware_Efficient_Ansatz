import os
from pathlib import Path
from typing import Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from pulser import Pulse, Register, Sequence
from pulser.devices import MockDevice
from pulser_diff.backend import TorchEmulator
from pulser_diff.derivative import deriv_param, deriv_time
from pulser_diff.utils import IMAT, ZMAT, kron
from pyqtorch.utils import SolverType
from sklearn.model_selection import train_test_split
from torch import Tensor, tensor
import optuna

from source.model import run_model_0
from source.train_loop import train_loop


def setup_model0_hparams(trial: optuna.trial.Trial, config: dict) -> dict:
    n_ancilliary_qubits = trial.suggest_int(
        "n_ancilliary_qubits", 0, config["max_ancilliary_qubits"]
    )
    sampling_rate = 0.6
    local_pulse_duration = trial.suggest_int("local_pulse_duration", 50, 130, step=10)
    global_pulse_duration = trial.suggest_int("global_pulse_duration", 50, 500, step=10)
    embed_pulse_duration = trial.suggest_int("embed_pulse_duration", 50, 130, step=10)
    # positions = np.random.uniform(
    #     -40, 40, size=(config["pca_components"] + n_ancilliary_qubits, 2)
    # )
    positions = []
    for atom in range(config["pca_components"] + n_ancilliary_qubits):
        x = trial.suggest_float(f"pos_x_{atom}", -40, 40)
        y = trial.suggest_float(f"pos_y_{atom}", -40, 40)
        positions.append([x, y])
    positions = np.array(positions, dtype=np.float32)
    positions = positions - np.mean(positions, axis=0)
    positions = torch.tensor(positions, requires_grad=True)

    local_pulses_omega = torch.tensor(
        [1.0] * (pca_components + n_ancilliary_qubits),
        dtype=torch.float32,
        requires_grad=True,
    )
    local_pulses_delta = torch.tensor(
        [0.5] * (pca_components + n_ancilliary_qubits),
        dtype=torch.float32,
        requires_grad=True,
    )
    global_pulse_omega = torch.tensor(0.7, dtype=torch.float32, requires_grad=True)
    global_pulse_delta = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    protocol = trial.suggest_categorical("protocol", ["min-delay", "wait-for-all"])

    out_params = {
        "n_ancilliary_qubits": n_ancilliary_qubits,
        "sampling_rate": sampling_rate,
        "local_pulse_duration": local_pulse_duration,
        "global_pulse_duration": global_pulse_duration,
        "embed_pulse_duration": embed_pulse_duration,
        "positions": positions,
        "local_pulses_omega": local_pulses_omega,
        "local_pulses_delta": local_pulses_delta,
        "global_pulse_omega": global_pulse_omega,
        "global_pulse_delta": global_pulse_delta,
        "data_save_file": config["data_save_file"],
        "lr": lr,
        "protocol": protocol,
    }
    print(f"Hyperparameters: {out_params}")

    return out_params


class Objective:
    """
    Objective class to use with optuna.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, trial: optuna.trial.Trial) -> float:
        return objective_(trial, self.config)


def setup_data_loaders(config):
    with h5py.File(config["filename_train"], "r") as f:
        X_train: np.ndarray = f["X_pca"][:n_load]  # type: ignore
        y_train: np.ndarray = f["y"][:n_load]  # type: ignore
    small_size = config["small_size"]
    train_kwargs = config["train_kwargs"]
    val_kwargs = config["test_kwargs"]
    test_kwargs = val_kwargs

    # only take items where y is 1 or 5
    mask = (y_train == 1) | (y_train == 5)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # convert y_train and y_test to 0 and 1 from 1 and 5
    y_train = y_train == 1  # .long()

    #   train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Create TensorDataset
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    # Create new data loaders with PCA-transformed data
    train_loader_pca = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader_pca = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    # smaller dataset with only a few samples
    small_train_dataset = torch.utils.data.TensorDataset(
        X_train[:small_size], y_train[:small_size]
    )
    small_train_loader = torch.utils.data.DataLoader(
        small_train_dataset, **train_kwargs
    )
    small_val_dataset = torch.utils.data.TensorDataset(
        X_val[:small_size], y_val[:small_size]
    )
    small_val_loader = torch.utils.data.DataLoader(small_val_dataset, **test_kwargs)

    del X_train, y_train, X_val, y_val  # free memory
    print(
        f"Train dataset: {len(train_dataset)}, validation dataset: {len(val_dataset)}"
    )

    return {
        "train_loader": small_train_loader,
        "val_loader": small_val_loader,
        "test_loader": None,
    }


def objective_(trial: optuna.trial.Trial, config) -> float:
    """
    Objective function to use with optuna. Trains a model with some random
    (from optuna chosen) hyperparameters and returns the test loss.
    """
    dls = setup_data_loaders(config)
    train_loader = dls["train_loader"]
    val_loader = dls["val_loader"]
    hparams = setup_model0_hparams(trial, config)

    train_properties, trained_params = train_loop(
        train_loader,
        val_loader,
        run_model_0,
        hparams,
        epochs=2,
        lr=hparams["lr"],
        data_save_file=hparams["data_save_file"],
    )

    final_loss = train_properties["val_loss"]
    final_accuracy = train_properties["val_accuracy"]
    print(f"Final validation loss: {final_loss:.4f}, accuracy: {final_accuracy:.4f}")

    # add accuracy to the trial
    trial.set_user_attr("final_accuracy", final_accuracy)
    trial.set_user_attr("train_accuracy_hist", train_properties["train_accuracy_hist"])
    trial.set_user_attr("train_loss_hist", train_properties["train_loss_hist"])
    trial.set_user_attr(
        "local_pulses_omega_hist", trained_params["local_pulses_omega_hist"]
    )
    trial.set_user_attr(
        "local_pulses_delta_hist", trained_params["local_pulses_delta_hist"]
    )
    trial.set_user_attr(
        "global_pulse_omega_hist", trained_params["global_pulse_omega_hist"]
    )
    trial.set_user_attr(
        "global_pulse_delta_hist", trained_params["global_pulse_delta_hist"]
    )
    trial.set_user_attr("positions_hist", trained_params["positions_hist"])

    return final_loss


if __name__ == "__main__":
    folder = Path("data") / "MNIST_PCA4"
    filename_train = folder / "mnist_pca4_train.h5"
    data_save_dir = Path("generated_data") / "4_pca_components" / "model0"
    data_save_file = data_save_dir / "output.csv"
    n_load = 32 * 32 * 30
    small_size = 16 * 16
    # small_size = 10
    batch_size = 16
    pca_components = 4
    optuna_log_db = "sqlite:///optuna/optuna_model0.db"

    train_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
    }
    test_kwargs = {
        "batch_size": 32,
        "shuffle": False,
        "num_workers": 1,
        "pin_memory": True,
    }
    config = {
        "filename_train": filename_train,
        "data_save_file": data_save_file,
        "n_load": n_load,
        "small_size": small_size,
        "pca_components": pca_components,
        "train_kwargs": train_kwargs,
        "test_kwargs": test_kwargs,
        "max_ancilliary_qubits": 2,  # maximum number of ancilliary qubits
    }

    study = optuna.create_study(
        direction="minimize",
        study_name="model0",
        storage=optuna_log_db,
        load_if_exists=True,
    )
    catches = (
        ValueError,
        RuntimeError,
    )
    study.optimize(
        Objective(config),
        n_trials=100,
        timeout=100_000,  # timeout in seconds
        show_progress_bar=True,
        gc_after_trial=True,
    )
