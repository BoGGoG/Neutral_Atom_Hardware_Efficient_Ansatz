"""
now with new model with learned output
"""

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
from torch import Tensor, is_inference, tensor
import optuna
from collections import Counter
import json
from source.NAHEA import NAHEA_nFeatures_BinClass_4
from source.trainer import Trainer


def undersample(X, y):
    classes = Counter(y)
    n_0 = classes[0]
    n_1 = classes[1]
    idxs_0 = np.where(y == 0)[0]
    idxs_1 = np.where(y == 1)[0]

    if n_0 < n_1:
        y_0 = y[idxs_0]
        X_0 = X[idxs_0]
        y_1 = y[idxs_1[: len(idxs_0)]]
        X_1 = X[idxs_1[: len(idxs_0)]]
        y = np.concatenate([y_0, y_1])
        X = np.concatenate([X_0, X_1])
    elif n_1 < n_0:
        y_0 = y[idxs_0[: len(idxs_1)]]
        X_0 = X[idxs_0[: len(idxs_1)]]
        y_1 = y[idxs_1]
        X_1 = X[idxs_1]
        y = np.concatenate([y_0, y_1])
        X = np.concatenate([X_0, X_1])

    # shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return X, y


def setup_data_loaders(config):
    n_load = config["n_load"]
    with h5py.File(config["filename_train"], "r") as f:
        X_train: np.ndarray = f["X"][:n_load]  # type: ignore
        y_train: np.ndarray = f["y"][:n_load]  # type: ignore
    small_size = config["small_size"]
    train_kwargs = config["train_kwargs"]
    val_kwargs = config["test_kwargs"]
    test_kwargs = val_kwargs

    # convert y_train and y_test to 0 and 1 from 1 and 5
    y_train = (y_train == 1).astype(np.float32)  # convert to float for BCELoss

    #   train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    X_train, y_train = undersample(X_train, y_train)
    X_val, y_val = undersample(X_val, y_val)

    # Create TensorDataset
    X_train = torch.tensor(X_train, dtype=torch.float32)[:, : config["pca_components"]]
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)[:, : config["pca_components"]]
    y_val = torch.tensor(y_val, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    # normalize X_train to [0, 1]
    X_train = (X_train - X_train.min(axis=0, keepdims=True)[0]) / (
        X_train.max(axis=0, keepdims=True)[0] - X_train.min(axis=0, keepdims=True)[0]
    )
    X_val = (X_val - X_val.min(axis=0, keepdims=True)[0]) / (
        X_val.max(axis=0, keepdims=True)[0] - X_val.min(axis=0, keepdims=True)[0]
    )

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
    print("loaded data")
    print(f"y_train classes: {Counter(np.array(y_train[:small_size]))}")
    print(f"y_val classes: {Counter(np.array(y_val[:small_size]))}")

    del X_train, y_train, X_val, y_val  # free memory
    print(
        f"Train dataset: {len(small_train_dataset)}, validation dataset: {len(small_val_dataset)}"
    )

    return {
        "train_loader": small_train_loader,
        "val_loader": small_val_loader,
        "test_loader": None,
    }


class Objective:
    """
    Objective class to use with optuna.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, trial: optuna.trial.Trial) -> float:
        return objective_(trial, self.config)


def setup_hparams(trial: optuna.trial.Trial, config: dict) -> dict:
    np.random.seed(trial.number)  # for reproducibility
    n_ancilliary_qubits = trial.suggest_int(
        "n_ancilliary_qubits", 0, config["max_ancilliary_qubits"]
    )
    pca_components = config["pca_components"]
    sampling_rate = config["sampling_rate"]
    local_pulse_duration = trial.suggest_int("local_pulse_duration", 50, 80, step=10)
    global_pulse_duration = trial.suggest_int("global_pulse_duration", 50, 150, step=10)
    embed_pulse_duration = trial.suggest_int("embed_pulse_duration", 50, 80, step=10)
    positions = []
    positions.append([3.4, 0.0])
    positions.append([-3.4, 0.0])
    if n_ancilliary_qubits == 1:
        positions.append([0.0, 4.2])
    if n_ancilliary_qubits == 2:
        positions.append([0.0, -4.2])
    if n_ancilliary_qubits > 2:
        raise ValueError(
            f"n_ancilliary_qubits should be 0, 1 or 2 for moons dataset, got {n_ancilliary_qubits}"
        )
    # for atom in range(config["pca_components"]):
    #     x = trial.suggest_float(f"pos_x_{atom}", -10, 10)
    #     # y = trial.suggest_float(f"pos_y_{atom}", -40, 40)
    #     y = 0.0
    #     positions.append([x, y])
    # for atom in range(n_ancilliary_qubits):
    #     x = trial.suggest_float(f"anc_pos_x_{atom}", -10, 10)
    #     y = trial.suggest_float(f"anc_pos_y_{atom}", -10, 10)
    #     positions.append([x, y])
    positions = np.array(positions, dtype=np.float32)
    # positions = positions - np.mean(positions, axis=0)

    local_pulses_omega_1 = [
        1.0 + np.random.uniform(-0.3, 0.3)
        for _ in range(pca_components + n_ancilliary_qubits)
    ]
    local_pulses_omega_2 = [
        1.0 + np.random.uniform(-0.3, 0.3)
        for _ in range(pca_components + n_ancilliary_qubits)
    ]
    local_pulses_delta_1 = [
        0.5 + np.random.uniform(-0.3, 0.3)
        for _ in range(pca_components + n_ancilliary_qubits)
    ]
    local_pulses_delta_2 = [
        0.5 + np.random.uniform(-0.3, 0.3)
        for _ in range(pca_components + n_ancilliary_qubits)
    ]
    # add some noise
    global_pulse_omega_1 = 1.0 + np.random.uniform(-0.3, 0.3)
    global_pulse_omega_2 = 1.0 + np.random.uniform(-0.3, 0.3)
    global_pulse_omega_3 = 1.0 + np.random.uniform(-0.3, 0.3)
    global_pulse_omega_4 = 1.0 + np.random.uniform(-0.3, 0.3)
    global_pulse_delta_1 = 0.5 + np.random.uniform(-0.3, 0.3)
    global_pulse_delta_2 = 0.5 + np.random.uniform(-0.3, 0.3)
    global_pulse_delta_3 = 0.5 + np.random.uniform(-0.3, 0.3)
    global_pulse_delta_4 = 0.5 + np.random.uniform(-0.3, 0.3)
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    protocol = trial.suggest_categorical("protocol", ["min-delay", "wait-for-all"])

    out_params = {
        "n_ancilliary_qubits": n_ancilliary_qubits,
        "sampling_rate": sampling_rate,
        "local_pulse_duration": local_pulse_duration,
        "global_pulse_duration": global_pulse_duration,
        "embed_pulse_duration": embed_pulse_duration,
        "positions": positions,
        "local_pulses_omega_1": local_pulses_omega_1,
        "local_pulses_omega_2": local_pulses_omega_2,
        "local_pulses_delta_1": local_pulses_delta_1,
        "local_pulses_delta_2": local_pulses_delta_2,
        "global_pulse_omega_1": global_pulse_omega_1,
        "global_pulse_omega_2": global_pulse_omega_2,
        "global_pulse_omega_3": global_pulse_omega_3,
        "global_pulse_omega_4": global_pulse_omega_4,
        "global_pulse_delta_1": global_pulse_delta_1,
        "global_pulse_delta_2": global_pulse_delta_2,
        "global_pulse_delta_3": global_pulse_delta_3,
        "global_pulse_delta_4": global_pulse_delta_4,
        "data_save_file": config["data_save_file"],
        "lr": lr,
        "protocol": protocol,
    }
    print(f"Hyperparameters: {out_params}")

    return out_params


def objective_(trial: optuna.trial.Trial, config) -> float:
    """
    Objective function to use with optuna. Trains a model with some random
    (from optuna chosen) hyperparameters and returns the test loss.
    """
    dls = setup_data_loaders(config)
    train_loader = dls["train_loader"]
    val_loader = dls["val_loader"]
    hparams_0 = setup_hparams(trial, config)
    lr = hparams_0["lr"]
    params = {  # could be simpler, but I like having a list of all parameters here
        "positions": hparams_0["positions"],
        "local_pulses_omega_1": hparams_0["local_pulses_omega_1"],
        "local_pulses_omega_2": hparams_0["local_pulses_omega_2"],
        "local_pulses_delta_1": hparams_0["local_pulses_delta_1"],
        "local_pulses_delta_2": hparams_0["local_pulses_delta_2"],
        "global_pulse_omega_1": hparams_0["global_pulse_omega_1"],
        "global_pulse_omega_2": hparams_0["global_pulse_omega_2"],
        "global_pulse_omega_3": hparams_0["global_pulse_omega_3"],
        "global_pulse_omega_4": hparams_0["global_pulse_omega_4"],
        "global_pulse_delta_1": hparams_0["global_pulse_delta_1"],
        "global_pulse_delta_2": hparams_0["global_pulse_delta_2"],
        "global_pulse_delta_3": hparams_0["global_pulse_delta_3"],
        "global_pulse_delta_4": hparams_0["global_pulse_delta_4"],
        "global_pulse_duration": hparams_0["global_pulse_duration"],
        "local_pulse_duration": hparams_0["local_pulse_duration"],
        "embed_pulse_duration": hparams_0["embed_pulse_duration"],
    }
    hparams = {
        "n_features": 2,
        "sampling_rate": hparams_0["sampling_rate"],
        "protocol": hparams_0["protocol"],
        "n_ancilliary_qubits": hparams_0["n_ancilliary_qubits"],
        "hidden_layers_dims": [],
        "output_dim": 1,  # output dimension for the final layer
    }
    model = NAHEA_nFeatures_BinClass_4(
        hparams=hparams, parameters=params, name="moons_model_4"
    )

    model.train()
    optimizer = torch.optim.Adam
    loss_fn = torch.nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        optimizer_kwargs={"lr": lr},
    )
    params = model.parameters()
    train_out = trainer.train(epochs=config["epochs"])
    val_losses = train_out["val_losses"]
    val_accs = train_out["val_accuracies"]
    train_losses = train_out["train_losses"]
    train_accs = train_out["train_accuracies"]

    model_params = model.parameters()
    # convert so they are json serializable
    params_for_saving = {
        param_name: param_value.tolist() for param_name, param_value in params.items()
    }
    trial.set_user_attr("model_params", params_for_saving)
    trial.set_user_attr("train_losses", train_losses)
    trial.set_user_attr("val_losses", val_losses)
    trial.set_user_attr("train_accuracies", [float(a) for a in train_accs])
    trial.set_user_attr("val_accuracies", [float(a) for a in val_accs])

    model_save_path = (
        config["run_dir"] / "saved_models" / ("model" + str(trial.number) + ".json")
    )
    os.makedirs(model_save_path.parent, exist_ok=True)
    model.save_state_dict(model_save_path)
    trial.set_user_attr("model_save_path", str(model_save_path))

    # todo:
    # - [ ] Tensorboard logging
    # - [ ] decision boundary plotting in tensorboard

    return val_losses[-1]  # return the last validation loss


if __name__ == "__main__":
    # Example configuration
    #
    run_dir = Path("runs") / "2025_07_10" / "moons_hyperopt"
    train_kwargs = {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 1,
        "pin_memory": False,
    }

    test_kwargs = {
        "batch_size": 32,
        "shuffle": False,
        "num_workers": 1,
        "pin_memory": True,
    }
    config = {
        "run_dir": run_dir,
        "epochs": 20,
        "n_load": 32 * 32 * 30,
        "small_size": 16 * 16,
        "pca_components": 2,
        "sampling_rate": 0.2,
        "max_ancilliary_qubits": 1,
        "filename_train": Path("data") / "moons" / "train.h5",
        "data_save_file": Path("generated_data")
        / "moons_2"
        / "NAHEA_nFeatures_BinClass_4"
        / "output.csv",
        "batch_size": 2,
        "train_kwargs": train_kwargs,
        "test_kwargs": test_kwargs,
    }

    # Create an Optuna study
    study = optuna.create_study(
        study_name="moons_hyperopt_4",
        direction="minimize",
        storage="sqlite:///"
        + "runs"
        + "/2025_07_10"
        + "/moons_hyperopt"
        + "/optuna.db",
        load_if_exists=True,
    )
    study.optimize(
        Objective(config),
        n_trials=100,
        timeout=100_000,  # 10,000 seconds = 10,000 / 360 = ~2.77 hours
    )

    # Print the best trial
    print("Best trial:")
    print(study.best_trial)
