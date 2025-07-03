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
from source.NAHEA import NAHEA_nFeatures_BinClass_2
from source.trainer import Trainer
from .moons_hyperopt_2 import setup_data_loaders


if __name__ == "__main__":
    # Create an Optuna study
    study = optuna.create_study(
        study_name="moons_hyperopt_2",
        direction="minimize",
        storage="sqlite:///"
        + "runs"
        + "/2025_07_02"
        + "/moons_hyperopt"
        + "/optuna.db",
        load_if_exists=True,
    )
    run_dir = Path("runs") / "2025_07_02" / "moons_hyperopt"
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
        # "n_load": 32 * 32 * 30,
        "n_load": None,
        "small_size": 1024,
        "pca_components": 2,
        "sampling_rate": 0.4,
        "max_ancilliary_qubits": 1,
        "filename_train": Path("data") / "moons" / "train.h5",
        "data_save_file": Path("generated_data")
        / "moons_2"
        / "NAHEA_nFeatures_BinClass_2"
        / "output.csv",
        "train_kwargs": train_kwargs,
        "test_kwargs": test_kwargs,
    }

    trial = study.get_trials()[1]
    print(f"{trial=}")
    model_save_path = trial.user_attrs["model_save_path"]
    lr = trial.params["lr"]
    print(f"{model_save_path=}")
    print(f"{lr=}")

    model = NAHEA_nFeatures_BinClass_2.from_file(model_save_path)
    print(f"{model}")

    dls = setup_data_loaders(config)
    train_loader = dls["train_loader"]
    val_loader = dls["val_loader"]

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
    train_out = trainer.train(epochs=config["epochs"])
    val_losses = train_out["val_losses"]
    val_accs = train_out["val_accuracies"]
    train_losses = train_out["train_losses"]
    train_accs = train_out["train_accuracies"]
    print("{train_losses[-1]=}")
    print("{train_accs[-1]=}")
    print("{val_losses[-1]=}")
    print("{val_accs[-1]=}")

    model_save_path = (
        config["run_dir"]
        / "saved_models"
        / ("model" + str(trial.number) + "_cont" + ".json")
    )
    model.save_state_dict(model_save_path)
