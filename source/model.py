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


def state_to_output(state: Tensor) -> Tensor:
    """
    Convert the final state of the system to a binary output.
    Here we assume that the output is 1 if the magnetization is positive,
    and 0 if it is negative.
    """
    magnetization = state.real.mean(dim=0)
    n_qubits = magnetization.shape[0]

    # Normalize the magnetization to [0, 1]
    out = (magnetization + n_qubits) / (2 * n_qubits)
    return out


def run_model_0(
    x: Tensor,
    positions: Tensor,
    local_pulses_omega: Tensor,
    local_pulses_delta: Tensor,
    global_pulse_omega: Tensor,
    global_pulse_delta: Tensor,
    global_pulse_duration=500,
    local_pulse_duration=250,
    embed_pulse_duration=250,
    sampling_rate=0.5,
    protocol: str = "min-delay",  #
    draw_reg_seq: bool = True,
) -> Tensor:
    """
    Run the model with the given parameters and return the output.
    Args:
        x (Tensor): Input data, PCA components.
        positions (Tensor): Positions of the qubits in the register.
        local_pulses_omega (Tensor): Amplitudes of the local pulses. (shape: (pca_components,))
        local_pulses_delta (Tensor): Detunings of the local pulses. (shape: (pca_components,))
        global_pulse_omega (Tensor): Amplitude of the global pulse. (shape: ())
        global_pulse_delta (Tensor): Detuning of the global pulse. (shape: ())
        protocol (str): Protocol to use for the sequence. Default is "min-delay", others: "no-delay", "wait-for-all"
    """
    assert len(positions) == len(local_pulses_omega) == len(local_pulses_delta)
    n_features = len(x)
    n_qubits = len(positions)
    n_ancilliary_qubits = len(positions) - len(x)

    reg = Register({"q" + str(i): pos for i, pos in enumerate(positions)})
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    for i in range(n_qubits):
        seq.declare_channel(f"rydberg_local_q{i}", "rydberg_local")
        seq.target(f"q{i}", channel=f"rydberg_local_q{i}")

    # make sure omegas are > 0
    global_pulse_omega = torch.abs(global_pulse_omega)
    local_pulses_omega = torch.abs(local_pulses_omega)

    # embed PCA components into the register through local pulses
    # x is already in the range [0, 1] due to normalization
    for i in range(n_features):
        pulse_local = Pulse.ConstantPulse(
            embed_pulse_duration, 1000 * x[i] * np.pi / embed_pulse_duration, 0.0, 0.0
        )  # Use x[i] as the amplitude
        seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)

    # global pulse
    pulse_global = Pulse.ConstantPulse(
        global_pulse_duration,
        global_pulse_omega * np.pi * 1000 / global_pulse_duration,
        global_pulse_delta * np.pi * 1000 / global_pulse_duration,
        0.0,
    )
    seq.add(pulse_global, "rydberg_global")
    seq.declare_variable("omega_global")
    seq.declare_variable("delta_global")

    # local pulses (including ancilliary qubits)
    for i in range(n_qubits):
        pulse_local = Pulse.ConstantPulse(
            local_pulse_duration,
            local_pulses_omega[i] * np.pi * 1000 / local_pulse_duration,
            local_pulses_delta[i] * np.pi * 1000 / local_pulse_duration,
            0.0,
        )
        seq.add(pulse_local, f"rydberg_local_q{i}", protocol="min-delay")
        seq.declare_variable(f"omega_q{i}")
        seq.declare_variable(f"delta_q{i}")

    # global pulse
    pulse_global = Pulse.ConstantPulse(
        global_pulse_duration,
        global_pulse_omega * np.pi * 1000 / global_pulse_duration,
        global_pulse_delta * np.pi * 1000 / global_pulse_duration,
        0.0,
    )
    seq.add(pulse_global, "rydberg_global")
    seq.declare_variable("omega1_global")
    seq.declare_variable("delta1_global")

    if draw_reg_seq:
        reg.draw(
            with_labels=True,
            draw_half_radius=True,
            blockade_radius=MockDevice.rydberg_blockade_radius(1.0),
        )
        seq.draw()

    sim = TorchEmulator.from_sequence(seq, sampling_rate=sampling_rate)
    results = sim.run(time_grad=True, dist_grad=True, solver=SolverType.DP5_SE)

    return state_to_output(results.states[-1]), results.states


def run_model_1(
    x: Tensor,
    positions: Tensor,
    local_pulses_omega: Tensor,
    local_pulses_delta: Tensor,
    global_pulse_omega: Tensor,
    global_pulse_delta: Tensor,
    global_pulse_duration=500,
    local_pulse_duration=250,
    embed_pulse_duration=250,
    sampling_rate=0.5,
    protocol: str = "min-delay",  #
    draw_reg_seq: bool = True,
) -> Tensor:
    """
    Run the model with the given parameters and return the output.
    Args:
        x (Tensor): Input data, PCA components.
        positions (Tensor): Positions of the qubits in the register.
        local_pulses_omega (Tensor): Amplitudes of the local pulses. (shape: (pca_components,))
        local_pulses_delta (Tensor): Detunings of the local pulses. (shape: (pca_components,))
        global_pulse_omega (Tensor): Amplitude of the global pulse. (shape: ())
        global_pulse_delta (Tensor): Detuning of the global pulse. (shape: ())
        protocol (str): Protocol to use for the sequence. Default is "min-delay", others: "no-delay", "wait-for-all"
    """
    assert len(positions) == len(local_pulses_omega) == len(local_pulses_delta)
    n_features = len(x)
    n_qubits = len(positions)
    n_ancilliary_qubits = len(positions) - len(x)

    reg = Register({"q" + str(i): pos for i, pos in enumerate(positions)})
    seq = Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    for i in range(n_qubits):
        seq.declare_channel(f"rydberg_local_q{i}", "rydberg_local")
        seq.target(f"q{i}", channel=f"rydberg_local_q{i}")

    # make sure omegas are > 0
    global_pulse_omega = torch.abs(global_pulse_omega)
    local_pulses_omega = torch.abs(local_pulses_omega)

    # embed PCA components into the register through local pulses
    # x is already in the range [0, 1] due to normalization
    for i in range(n_features):
        pulse_local = Pulse.ConstantPulse(
            embed_pulse_duration, 1000 * x[i] * np.pi / embed_pulse_duration, 0.0, 0.0
        )  # Use x[i] as the amplitude
        seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)

    # global pulse
    pulse_global = Pulse.ConstantPulse(
        global_pulse_duration,
        global_pulse_omega * np.pi * 1000 / global_pulse_duration,
        global_pulse_delta * np.pi * 1000 / global_pulse_duration,
        0.0,
    )
    seq.add(pulse_global, "rydberg_global")
    seq.declare_variable("omega_global")
    seq.declare_variable("delta_global")

    # local pulses (including ancilliary qubits)
    for i in range(n_qubits):
        pulse_local = Pulse.ConstantPulse(
            local_pulse_duration,
            local_pulses_omega[i] * np.pi * 1000 / local_pulse_duration,
            local_pulses_delta[i] * np.pi * 1000 / local_pulse_duration,
            0.0,
        )
        seq.add(pulse_local, f"rydberg_local_q{i}", protocol="min-delay")
        seq.declare_variable(f"omega_q{i}")
        seq.declare_variable(f"delta_q{i}")

    # global pulse
    pulse_global = Pulse.ConstantPulse(
        global_pulse_duration,
        global_pulse_omega * np.pi * 1000 / global_pulse_duration,
        global_pulse_delta * np.pi * 1000 / global_pulse_duration,
        0.0,
    )
    seq.add(pulse_global, "rydberg_global")
    seq.declare_variable("omega1_global")
    seq.declare_variable("delta1_global")

    # local pulses (data reuploading)
    for i in range(n_features):
        pulse_local = Pulse.ConstantPulse(
            embed_pulse_duration, 1000 * x[i] * np.pi / embed_pulse_duration, 0.0, 0.0
        )  # Use x[i] as the amplitude
        seq.add(pulse_local, f"rydberg_local_q{i}", protocol="min-delay")
        seq.declare_variable(f"omega2_q{i}")
        seq.declare_variable(f"delta2_q{i}")

    # global pulse
    pulse_global = Pulse.ConstantPulse(
        global_pulse_duration,
        global_pulse_omega * np.pi * 1000 / global_pulse_duration,
        global_pulse_delta * np.pi * 1000 / global_pulse_duration,
        0.0,
    )
    seq.add(pulse_global, "rydberg_global")
    seq.declare_variable("omega2_global")
    seq.declare_variable("delta2_global")

    # local pulses (including ancilliary qubits)
    for i in range(n_qubits):
        pulse_local = Pulse.ConstantPulse(
            local_pulse_duration,
            local_pulses_omega[i] * np.pi * 1000 / local_pulse_duration,
            local_pulses_delta[i] * np.pi * 1000 / local_pulse_duration,
            0.0,
        )
        seq.add(pulse_local, f"rydberg_local_q{i}", protocol="min-delay")
        seq.declare_variable(f"omega3_q{i}")
        seq.declare_variable(f"delta3_q{i}")

    # global pulse
    pulse_global = Pulse.ConstantPulse(
        global_pulse_duration,
        global_pulse_omega * np.pi * 1000 / global_pulse_duration,
        global_pulse_delta * np.pi * 1000 / global_pulse_duration,
        0.0,
    )
    seq.add(pulse_global, "rydberg_global")
    seq.declare_variable("omega3_global")
    seq.declare_variable("delta3_global")

    if draw_reg_seq:
        reg.draw(
            with_labels=True,
            draw_half_radius=True,
            blockade_radius=MockDevice.rydberg_blockade_radius(1.0),
        )
        seq.draw()

    sim = TorchEmulator.from_sequence(seq, sampling_rate=sampling_rate)
    results = sim.run(time_grad=True, dist_grad=True, solver=SolverType.DP5_SE)

    return state_to_output(results.states[-1]), results.states


if __name__ == "__main__":
    # folder = Path("data") / "MNIST_PCA4"
    # filename_train = folder / "mnist_pca4_train.h5"
    n_load = 32 * 32 * 20
    small_size = 32 * 32
    pca_components = 4

    train_kwargs = {
        "batch_size": 32,
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

    # local_pulses_omega = torch.tensor( [1.0] * pca_components, dtype=torch.float32, requires_grad=True)
    # local_pulses_delta = torch.tensor( [0.5] * pca_components, dtype=torch.float32, requires_grad=True)
    # global_pulse_omega = torch.tensor(0.7, dtype=torch.float32, requires_grad=True)
    # global_pulse_delta = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
    local_pulses_omega = tensor(
        [0.8641, 1.2907, 1.2563, 0.8885, 0.5], requires_grad=True
    )
    local_pulses_delta = tensor(
        [0.7074, 0.2148, 0.4913, 0.8155, 0.5], requires_grad=True
    )
    global_pulse_omega = tensor(0.6797, requires_grad=True)
    global_pulse_delta = tensor(0.5361, requires_grad=True)

    print("Running model")
    out, states = run_model_1(
        small_train_dataset[0][0],
        positions,
        local_pulses_omega,
        local_pulses_delta,
        global_pulse_omega,
        global_pulse_delta,
        local_pulse_duration=local_pulse_duration,
        global_pulse_duration=global_pulse_duration,
        embed_pulse_duration=embed_pulse_duration,
        draw_reg_seq=False,
        sampling_rate=sampling_rate,
        protocol="min-delay",  # "wait-for-all", "no-delay", "min-delay"
    )
