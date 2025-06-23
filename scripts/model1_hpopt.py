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

from source.model import run_model_1
from source.train_loop import train_loop

if __name__ == "__main__":
    folder = Path("data") / "MNIST_PCA4"
    filename_train = folder / "mnist_pca4_train.h5"
    n_load = 32 * 32 * 10
    # small_size = 32 * 32
    small_size = 20
    batch_size = 8
    pca_components = 4

    train_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
    }
    test_kwargs = {
        "batch_size": 1024,
        "shuffle": False,
        "num_workers": 1,
        "pin_memory": True,
    }

    print("Loading data...")
    with h5py.File(filename_train, "r") as f:
        X_train: np.ndarray = f["X_pca"][:n_load]  # type: ignore
        y_train: np.ndarray = f["y"][:n_load]  # type: ignore
    print("Data loaded.")

    print(X_train.shape, y_train.shape)

    # only take items where y is 1 or 5
    mask = (y_train == 1) | (y_train == 5)
    X_train = X_train[mask]
    y_train = y_train[mask]
    print(X_train.shape, y_train.shape)

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

    x_sample = small_train_dataset[0][0]
    y_sample = small_train_dataset[0][1]
    print(f"Sample PCA data point: {x_sample}")
    print(f"Sample label: {y_sample.item()}")

    #####################################################
    # Start training the model
    #####################################################
    np.random.seed(42)  # Keeps results consistent between runs

    n_ancilliary_qubits = 1
    sampling_rate = 0.5
    local_pulse_duration = 50
    global_pulse_duration = 50
    embed_pulse_duration = 50
    positions = np.random.randint(0, 38, size=(pca_components + n_ancilliary_qubits, 2))
    # positions = np.array(
    #     [
    #         [10.60559012, -0.95824388],
    #         [-9.9838079, 4.71348483],
    #         [4.0438376, 4.99340997],
    #         [-3.75676635, -7.58706094],
    #         # below are ancilliary qubits
    #         [-15.0, -5.0],
    #     ]
    # )
    positions = positions - np.mean(positions, axis=0)
    positions = torch.tensor(positions, requires_grad=True)
    print("start positions:", positions)

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
    # local_pulses_omega = tensor(
    #     [0.8641, 1.2907, 1.2563, 0.8885, 0.5], requires_grad=True
    # )
    # local_pulses_delta = tensor(
    #     [0.7074, 0.2148, 0.4913, 0.8155, 0.5], requires_grad=True
    # )
    # global_pulse_omega = tensor(0.6797, requires_grad=True)
    # global_pulse_delta = tensor(0.5361, requires_grad=True)

    print("Training model")
    params_dic = {
        "positions": positions,
        "local_pulses_omega": local_pulses_omega,
        "local_pulses_delta": local_pulses_delta,
        "global_pulse_omega": global_pulse_omega,
        "global_pulse_delta": global_pulse_delta,
        "local_pulse_duration": local_pulse_duration,
        "global_pulse_duration": global_pulse_duration,
        "embed_pulse_duration": embed_pulse_duration,
        "sampling_rate": sampling_rate,
    }
    data_save_dir = Path("generated_data") / "4_pca_components" / "2"
    data_save_file = data_save_dir / "output.csv"

    train_properties, trained_params = train_loop(
        small_train_loader,
        small_val_loader,
        run_model_1,
        params_dic,
        epochs=1,
        # positions,
        # local_pulses_omega,
        # local_pulses_delta,
        # global_pulse_omega,
        # global_pulse_delta,
        # local_pulse_duration=local_pulse_duration,
        # global_pulse_duration=global_pulse_duration,
        # embed_pulse_duration=embed_pulse_duration,
        # sampling_rate=sampling_rate,
        # n_epochs=1000,  # 1000
        # n_ancilliary_qubits=n_ancilliary_qubits,
    )
    print("Training complete.")
    print("Validation:")
    print(f"Loss: {train_properties['val_loss']:.4f}")
    print(f"Accuracy: {train_properties['val_accuracy']:.4f}")

    # out, states = run_model_1(
    #     small_train_dataset[0][0],
    #     positions,
    #     local_pulses_omega,
    #     local_pulses_delta,
    #     global_pulse_omega,
    #     global_pulse_delta,
    #     local_pulse_duration=local_pulse_duration,
    #     global_pulse_duration=global_pulse_duration,
    #     embed_pulse_duration=embed_pulse_duration,
    #     draw_reg_seq=False,
    #     protocol="min-delay",  # "wait-for-all", "no-delay", "min-delay"
    # )
