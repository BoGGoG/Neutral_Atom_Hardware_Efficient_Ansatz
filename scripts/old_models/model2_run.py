"""
Neutral atoms model working on PCA components.
Run from root directory with: `python -m source.model`
"""

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
from source.model import run_model_2
from scripts.model2_hpopt import setup_data_loaders
import os
import optuna
from source.train_loop import train_loop
import pickle

if __name__ == "__main__":
    folder = Path("data") / "MNIST_PCA4"
    filename_train = folder / "mnist_pca4_train.h5"
    data_save_dir = Path("generated_data") / "2_pca_components" / "2_run"
    os.makedirs(data_save_dir, exist_ok=True)
    data_save_file = data_save_dir / "output.csv"
    n_load = 32 * 32 * 30
    small_size = 32 * 32
    epochs = 10
    # small_size = 10
    batch_size = 32
    pca_components = 2
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
        "max_ancilliary_qubits": 0,  # maximum number of ancilliary qubits
    }
    dls = setup_data_loaders(config)
    ds_val = dls["val_loader"].dataset

    optuna_log_db = "sqlite:///optuna/optuna_model2.db"
    study_loaded = optuna.load_study(study_name="model2", storage=optuna_log_db)
    print(f"{len(study_loaded.trials)=}")
    trial_nr = 4
    trial = study_loaded.trials[trial_nr]
    print(f"{trial.user_attrs['trained_params'].keys()=}")
    trained_params = {
        "local_pulses_omega": trial.user_attrs["trained_params"]["local_pulses_omega"][
            -1
        ],
        "local_pulses_delta": trial.user_attrs["trained_params"]["local_pulses_delta"][
            -1
        ],
        "global_pulse_omega": trial.user_attrs["trained_params"]["global_pulse_omega"][
            -1
        ],
        "global_pulse_delta": trial.user_attrs["trained_params"]["global_pulse_delta"][
            -1
        ],
        "positions": trial.user_attrs["trained_params"]["positions"][-1],
    }
    hparams = trial.params
    print(hparams)

    # x_example = ds_val[0][0]
    # run_model_2(
    #     x_example,
    #     positions=trained_params["positions"],
    #     local_pulses_omega=Tensor(trained_params["local_pulses_omega"]),
    #     local_pulses_delta=Tensor(trained_params["local_pulses_delta"]),
    #     global_pulse_omega=Tensor([trained_params["global_pulse_omega"]]),
    #     global_pulse_delta=trained_params["global_pulse_delta"],
    #     local_pulse_duration=hparams["local_pulse_duration"],
    #     global_pulse_duration=hparams["global_pulse_duration"],
    #     embed_pulse_duration=hparams["embed_pulse_duration"],
    #     draw_reg_seq=True,
    #     draw_only=True,
    # )

    print("Positions:", trained_params["positions"])

    params = {
        "n_ancilliary_qubits": hparams["n_ancilliary_qubits"],
        "sampling_rate": 0.4,
        "local_pulse_duration": hparams["local_pulse_duration"],
        "global_pulse_duration": hparams["global_pulse_duration"],
        "embed_pulse_duration": hparams["embed_pulse_duration"],
        "positions": tensor(trained_params["positions"], requires_grad=True),
        "local_pulses_omega": tensor(
            trial.user_attrs["trained_params"]["local_pulses_omega"][-1],
            requires_grad=True,
        ),
        "local_pulses_delta": tensor(
            trial.user_attrs["trained_params"]["local_pulses_delta"][-1],
            requires_grad=True,
        ),
        "global_pulse_omega": tensor(
            trial.user_attrs["trained_params"]["global_pulse_omega"][-1],
            requires_grad=True,
        ),
        "global_pulse_delta": tensor(
            trial.user_attrs["trained_params"]["global_pulse_delta"][-1],
            requires_grad=True,
        ),
        "data_save_file": data_save_file,
        "protocol": hparams["protocol"],
    }

    train_properties, trained_params = train_loop(
        dls["train_loader"],
        dls["val_loader"],
        run_model_2,
        params,
        epochs=epochs,
        lr=hparams["lr"],
    )

    final_params = {
        "n_ancilliary_qubits": hparams["n_ancilliary_qubits"],
        "sampling_rate": 0.4,
        "local_pulse_duration": hparams["local_pulse_duration"],
        "global_pulse_duration": hparams["global_pulse_duration"],
        "embed_pulse_duration": hparams["embed_pulse_duration"],
        "positions": trained_params["positions"][-1],
        "local_pulses_omega": trained_params["local_pulses_omega"][-1],
        "local_pulses_delta": trained_params["local_pulses_delta"][-1],
        "global_pulse_omega": trained_params["global_pulse_omega"][-1],
        "global_pulse_delta": trained_params["global_pulse_delta"][-1],
        "data_save_file": data_save_file,
        "protocol": hparams["protocol"],
        "train_loss_hist": train_properties["train_loss_hist"],
        "train_accuracy_hist": train_properties["train_accuracy_hist"],
        "val_loss": train_properties["val_loss"],
        "val_accuracy": train_properties["val_accuracy"],
    }
    save_dir = Path("runs", "model2", "run_0")
    final_params_save_file = save_dir / "run_0.pickle"
    os.makedirs(save_dir, exist_ok=True)
    with open(final_params_save_file, "wb") as f:
        pickle.dump(final_params, f)
        print(f"Saved final_params to {final_params_save_file}")

    training_history_save_file = save_dir / "train_history_0.pickle"
    save_dic = {
        "train_properties": train_properties,
        "train_history": trained_params,
    }
    with open(training_history_save_file, "wb") as f:
        pickle.dump(save_dic, f)
        print(f"Saved training history to {training_history_save_file}")

    print(trained_params)
    train_accuracy_hist = train_properties["train_accuracy_hist"]
    plt.plot(train_accuracy_hist)
    plt.show()


# def run_model_2(
#     x: Tensor,
#     positions: Tensor,
#     local_pulses_omega: Tensor,
#     local_pulses_delta: Tensor,
#     global_pulse_omega: Tensor,
#     global_pulse_delta: Tensor,
#     global_pulse_duration=500,
#     local_pulse_duration=250,
#     embed_pulse_duration=250,
#     sampling_rate=0.5,
#     protocol: str = "min-delay",  #
#     draw_reg_seq: bool = True,
#     draw_only: bool = False,
# ) -> Tensor:
