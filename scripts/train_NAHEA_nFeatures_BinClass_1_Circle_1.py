from source.NAHEA import NAHEA, NAHEA_nFeatures_BinClass_1
import torch
import pytest
from source.trainer import Trainer, undersample
from pathlib import Path
import os
import copy
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


def setup_data_loaders(config):
    n_load = config["n_load"]
    with h5py.File(config["filename_train"], "r") as f:
        X_train: np.ndarray = f["X"][:n_load]  # type: ignore
        y_train: np.ndarray = f["y"][:n_load]  # type: ignore
    small_size = config["small_size"]
    train_kwargs = config["train_kwargs"]
    val_kwargs = config["test_kwargs"]
    test_kwargs = val_kwargs

    # only take items where y is 1 or 5
    mask = (y_train == 0) | (y_train == 1)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # convert y_train and y_test to 0 and 1 from 1 and 5
    y_train = y_train == 1  # .long()

    #   train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    X_train, y_train = undersample(X_train, y_train)
    X_val, y_val = undersample(X_val, y_val)

    # Create TensorDataset
    X_train = torch.tensor(X_train, dtype=torch.double)[:, : config["pca_components"]]
    y_train = torch.tensor(y_train, dtype=torch.double)

    # scale the data to 0-1 range
    X_train = (X_train - X_train.min(axis=0, keepdims=True)[0]) / (
        X_train.max(axis=0, keepdims=True)[0] - X_train.min(axis=0, keepdims=True)[0]
    )
    X_val = (X_val - X_val.min(axis=0, keepdims=True)[0]) / (
        X_val.max(axis=0, keepdims=True)[0] - X_val.min(axis=0, keepdims=True)[0]
    )

    X_val = torch.tensor(X_val, dtype=torch.double)[:, : config["pca_components"]]
    y_val = torch.tensor(y_val, dtype=torch.double)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

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


def setup_model():
    sampling_rate = 0.05
    hparams = {
        "n_features": 2,
        "sampling_rate": sampling_rate,
        "protocol": "min-delay",
        "n_ancilliary_qubits": (n_ancilliary_qubits := 1),
    }
    parameters = {
        "positions": [[-3.6672354, 0.0], [3.6672359, 0.0], [0.0, 3.6672359]],
        "local_pulses_omega": [1.1559689, 1.6583259, 1.0],
        "local_pulses_delta": [-0.76122487, 1.5434982, 0.0],
        "global_pulse_omega": -0.26719406,
        "global_pulse_delta": 1.0807998,
        "global_pulse_duration": 50,
        "local_pulse_duration": 50,
        "embed_pulse_duration": 80,
    }
    folder = Path("data") / "circle"
    filename_train = folder / "train.h5"
    epochs = 50
    n_load = 32 * 32 * 30
    small_size = 16 * 16
    # small_size = 3
    batch_size = 16
    pca_components = 2
    logging_dir = Path("logs") / "NAHEA" / "circle"
    optuna_log_db = logging_dir / "circle.db"
    # optuna_log_db = "sqlite:///optuna/optuna_model2.db"
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
    train_config = {
        "filename_train": filename_train,
        "data_save_file": Path("generated_data")
        / "2_pca_components"
        / "NAHEA_nFeatures_BinClass_1_circle_2"
        / "output.csv",
        "n_load": n_load,
        "small_size": small_size,
        "pca_components": pca_components,
        "train_kwargs": train_kwargs,
        "test_kwargs": test_kwargs,
        "epochs": epochs,
        "sampling_rate": sampling_rate,
    }

    model = NAHEA_nFeatures_BinClass_1(
        hparams=hparams, parameters=parameters, name="test_model_2features"
    )
    return model, train_config


if __name__ == "__main__":
    rundir = Path("runs") / "2025-07-01" / "NAHEA_nFeatures_BinClass_1_circle_2"
    os.makedirs(rundir, exist_ok=True)
    model_save_path = rundir / "model.json"
    model, train_config = setup_model()
    dls = setup_data_loaders(train_config)
    train_loader = dls["train_loader"]
    val_loader = dls["val_loader"]

    lr = 0.01
    model.train()
    optimizer = torch.optim.Adam
    loss_fn = torch.nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    trainer = Trainer(
        model,
        optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_kwargs={"lr": lr},
    )

    trainer.train(epochs=train_config["epochs"])

    model.save_state_dict(model_save_path)
