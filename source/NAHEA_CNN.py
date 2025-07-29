import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pulser import Pulse, Register, Sequence
from pulser.devices import MockDevice
from pulser_diff.backend import TorchEmulator
from pyqtorch.utils import SolverType
from torch import Tensor
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .NAHEA import NAHEA
import torch.nn as nn


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


def state_to_output_learned(state: Tensor, weights: Tensor) -> Tensor:
    """
    Convert the final state of the system to a binary output.
    Here we assume the output is a weighted sum of the state components.
    $\langle O \rangle = \sum_i w_i \langle \sigma^z_i \rangle$,
    """
    # for each state, take the absolute value of the complex coefficient
    coeffs = state.abs().squeeze()
    out = coeffs @ weights

    return out


class NAHEA_CNN_1(NAHEA):
    """Attempt at NAHEA CNN"""

    def __init__(
        self,
        hparams: dict,
        parameters: dict,
        name: str = "NAHEA_CNN_1 model",
    ):
        """ """
        self.parameters_required_keys = [
            "positions",
            "local_pulses_omega_1",
            # "local_pulses_omega_2",
            "local_pulses_delta_1",
            # "local_pulses_delta_2",
            "global_pulse_omega_1",
            "global_pulse_omega_2",
            # "global_pulse_omega_3",
            # "global_pulse_omega_4",
            "global_pulse_delta_1",
            "global_pulse_delta_2",
            # "global_pulse_delta_3",
            # "global_pulse_delta_4",
            "global_pulse_duration",
            "local_pulse_duration",
            "embed_pulse_duration",
        ]
        self.non_trainable_params = [
            "global_pulse_duration",
            "local_pulse_duration",
            "embed_pulse_duration",
        ]
        self.hparams_required_keys = [
            "n_features",
            "sampling_rate",
            "protocol",
            "n_ancilliary_qubits",
            "hidden_layers_dims",
        ]
        # random initialization of parameters
        self.input_length = hparams.get("input_length")
        self.stride = hparams["stride"]
        self.seq_len = hparams["input_length"]
        self.conv_out_len = len(
            range(0, self.input_length - hparams["n_features"] + 1, hparams["stride"])
        )
        self.classical_params = {
            "conv_params": (
                torch.randn(size=[2 ** hparams["n_features"]], dtype=torch.float64)
                * 0.5
                + 1
            )
            / hparams["n_features"] ** 2,
        }

        # set up final FC NN
        hidden_layers_dims = hparams.get("hidden_layers_dims", [])
        self.fc_final = nn.Sequential()
        # output length of the convolution
        current_dim = len(
            range(0, self.seq_len - hparams["n_features"] + 1, self.stride)
        )
        for dim in hidden_layers_dims:
            self.fc_final.append(nn.Linear(current_dim, dim, dtype=torch.float64))
            self.fc_final.append(nn.ReLU())
            current_dim = dim
        self.output_dim = hparams["output_dim"]
        self.fc_final.append(
            nn.Linear(current_dim, self.output_dim, dtype=torch.float64)
        )

        ### ---
        super().__init__(hparams, parameters, name)
        self.input_checks()

        for key in self.classical_params:
            self.classical_params[key].requires_grad = True
            self._parameters[key] = self.classical_params[key]

        # initialize self.fc_final parameters
        for name, param in self.fc_final.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

        # add fc parameters to _parameters
        for name, param in self.fc_final.named_parameters():
            print(f"Adding parameter {name} to model")
            self._parameters[name] = param

    def input_checks(self):
        assert (
            len(self._parameters["positions"])
            == len(self._parameters["local_pulses_omega_1"])
            == len(self._parameters["local_pulses_delta_1"])
        )

    def setup_register(self):
        """
        First n_features qubits are the input features,
        remaining qubits are ancilliary qubits.
        ToDo:
        - embedding pulses delta?
        - I wanted to generate the register as a parameterized sequence, but as far as I can tell, the
        """
        positions = self._parameters["positions"]
        n_qubits = len(positions)
        # make sure omegas are > 0. Maybe this could be done somewhere else, but when training, this could also happen and I don't want a crash.
        global_pulse_omega_1 = torch.abs(self._parameters["global_pulse_omega_1"])
        global_pulse_omega_2 = torch.abs(self._parameters["global_pulse_omega_2"])
        # global_pulse_omega_3 = torch.abs(self._parameters["global_pulse_omega_3"])
        # global_pulse_omega_4 = torch.abs(self._parameters["global_pulse_omega_4"])
        global_pulse_delta_1 = self._parameters["global_pulse_delta_1"]
        global_pulse_delta_2 = self._parameters["global_pulse_delta_2"]
        # global_pulse_delta_3 = self._parameters["global_pulse_delta_3"]
        # global_pulse_delta_4 = self._parameters["global_pulse_delta_4"]
        local_pulses_omega_1 = torch.abs(self._parameters["local_pulses_omega_1"])
        # local_pulses_omega_2 = torch.abs(self._parameters["local_pulses_omega_2"])
        global_pulse_delta_1 = self._parameters["global_pulse_delta_1"]
        # global_pulse_delta_2 = self._parameters["global_pulse_delta_2"]
        local_pulses_delta_1 = self._parameters["local_pulses_delta_1"]
        # local_pulses_delta_2 = self._parameters["local_pulses_delta_2"]
        global_pulse_duration = self._parameters["global_pulse_duration"]
        local_pulse_duration = self._parameters["local_pulse_duration"]
        embed_pulse_duration = self._parameters["embed_pulse_duration"]
        n_features = self.hparams["n_features"]
        protocol = self.hparams["protocol"]

        reg = Register({"q" + str(i): pos for i, pos in enumerate(positions)})
        seq = Sequence(reg, MockDevice)
        x = seq.declare_variable("x", dtype=float, size=n_features)

        seq.declare_channel("rydberg_global", "rydberg_global")
        for i in range(n_qubits):
            seq.declare_channel(f"rydberg_local_q{i}", "rydberg_local")
            seq.target(f"q{i}", channel=f"rydberg_local_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega_1 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta_1 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # embedding pulses
        # x is already in the range [0, 1] due to normalization
        for i in range(n_features):
            pulse_local = Pulse.ConstantPulse(
                embed_pulse_duration,
                1000 * x[i] * np.pi / embed_pulse_duration,
                0.0,
                0.0,  # pyright: ignore
            )  # Use x[i] as the amplitude
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)

        # local pulses (including ancilliary qubits)
        for i in range(n_qubits):
            pulse_local = Pulse.ConstantPulse(
                local_pulse_duration,
                local_pulses_omega_1[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                local_pulses_delta_1[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                0.0,
            )
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)
            # seq.declare_variable(f"omega_q{i}")
            # seq.declare_variable(f"delta_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega_2 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta_2 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        return seq

    def forward(
        self,
        x: Tensor,
        time_grad: bool = False,
        dist_grad: bool = False,
        solver: str = "DP5_SE",
    ) -> dict:
        """
        parameters
        - x: Tensor
        - time_grad: bool, whether to store the gradients for all evaluation times, allowing derivatives w/r to these times
        - dist_grad: bool, allowes calculation for derivatives w/r to the inter-qubit distances r_ij
        - solver: SolverType, the solver to use for the simulation

        """
        if solver == "DP5_SE":
            solver = SolverType.DP5_SE
        elif solver == "KRYLOV_SE":
            solver = SolverType.KRYLOV_SE
        out = {"results": [], "output": []}
        seq_len = x.shape[0]
        n_features = self.hparams["n_features"]
        stride = self.hparams.get("stride", 3)

        for pos in range(0, seq_len - n_features + 1, stride):
            x_slice = x[pos : pos + n_features]
            base_seq = self.setup_register()
            seq_built = base_seq.build(x=x_slice)
            sampling_rate = self.hparams["sampling_rate"]
            sim = TorchEmulator.from_sequence(seq_built, sampling_rate=sampling_rate)
            if self.training:
                results = sim.run(
                    time_grad=time_grad, dist_grad=dist_grad, solver=SolverType.DP5_SE
                )
            else:
                with torch.no_grad():
                    results = sim.run(
                        time_grad=False, dist_grad=False, solver=SolverType.DP5_SE
                    )
            output = state_to_output_learned(
                results.states[-1], self.classical_params["conv_params"]
            ).squeeze()
            out["output"].append(output)
            out["results"].append(results)
        out["output"] = torch.stack(out["output"])
        out["output"] = self.fc_final(out["output"])

        return out


class NAHEA_CNN_2(NAHEA):
    """With learned embedding before pulse"""

    def __init__(
        self,
        hparams: dict,
        parameters: dict,
        name: str = "NAHEA_CNN_2 model",
    ):
        """ """
        self.parameters_required_keys = [
            "positions",
            "local_pulses_omega_1",
            # "local_pulses_omega_2",
            "local_pulses_delta_1",
            # "local_pulses_delta_2",
            "global_pulse_omega_1",
            "global_pulse_omega_2",
            # "global_pulse_omega_3",
            # "global_pulse_omega_4",
            "global_pulse_delta_1",
            "global_pulse_delta_2",
            # "global_pulse_delta_3",
            # "global_pulse_delta_4",
            "global_pulse_duration",
            "local_pulse_duration",
            "embed_pulse_duration",
        ]
        self.non_trainable_params = [
            "global_pulse_duration",
            "local_pulse_duration",
            "embed_pulse_duration",
        ]
        self.hparams_required_keys = [
            "sampling_rate",
            "protocol",
            "n_ancilliary_qubits",
            "hidden_layers_dims",
        ]
        self.input_length = hparams.get("input_length")
        self.stride = hparams["stride"]
        self.seq_len = hparams["input_length"]
        self.n_qubits = len(parameters["positions"])
        self.conv_out_len = len(
            range(0, self.input_length - hparams["kernel_size"] + 1, hparams["stride"])
        )
        self.classical_params = {
            "conv_params": (
                torch.randn(size=[2**self.n_qubits], dtype=torch.float64) * 0.5 + 1
            )
            / self.n_qubits**2,
        }
        # embedding FC
        self.embedding_FC = nn.Sequential()
        current_dim = hparams["kernel_size"]  # kernel size
        for dim in hparams["embedding_FC_hidden_dims"]:
            self.embedding_FC.append(nn.Linear(current_dim, dim, dtype=torch.float64))
            self.embedding_FC.append(nn.ReLU())
            current_dim = dim
        self.embedding_FC.append(
            nn.Linear(current_dim, self.n_qubits, dtype=torch.float64)
        )
        self.embedding_FC.append(nn.Sigmoid())  # seq needs x in [0, 1]

        # set up final FC NN
        hidden_layers_dims = hparams.get("hidden_layers_dims", [])
        self.fc_final = nn.Sequential()
        # output length of the convolution
        current_dim = len(
            range(0, self.seq_len - hparams["kernel_size"] + 1, self.stride)
        )
        for dim in hidden_layers_dims:
            self.fc_final.append(nn.Linear(current_dim, dim, dtype=torch.float64))
            self.fc_final.append(nn.ReLU())
            current_dim = dim
        self.output_dim = hparams["output_dim"]
        self.fc_final.append(
            nn.Linear(current_dim, self.output_dim, dtype=torch.float64)
        )

        ### ---
        super().__init__(hparams, parameters, name)
        self.input_checks()

        for key in self.classical_params:
            self.classical_params[key].requires_grad = True
            self._parameters[key] = self.classical_params[key]

        # initialize self.fc_final parameters
        for name, param in self.fc_final.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

        # initialize self.embedding_FC parameters
        for name, param in self.embedding_FC.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

        # add fc parameters to _parameters
        for name, param in self.fc_final.named_parameters():
            self._parameters[name] = param

        # add self.embedding_FC of embedding_FC to _parameters
        for name, param in self.embedding_FC.named_parameters():
            self._parameters["embedding_fc_" + name] = param

    def input_checks(self):
        assert (
            len(self._parameters["positions"])
            == len(self._parameters["local_pulses_omega_1"])
            == len(self._parameters["local_pulses_delta_1"])
        )

    def setup_register(self):
        """
        First n_features qubits are the input features,
        remaining qubits are ancilliary qubits.
        ToDo:
        - embedding pulses delta?
        - I wanted to generate the register as a parameterized sequence, but as far as I can tell, the
        """
        positions = self._parameters["positions"]
        n_qubits = len(positions)
        # make sure omegas are > 0. Maybe this could be done somewhere else, but when training, this could also happen and I don't want a crash.
        global_pulse_omega_1 = torch.abs(self._parameters["global_pulse_omega_1"])
        global_pulse_omega_2 = torch.abs(self._parameters["global_pulse_omega_2"])
        # global_pulse_omega_3 = torch.abs(self._parameters["global_pulse_omega_3"])
        # global_pulse_omega_4 = torch.abs(self._parameters["global_pulse_omega_4"])
        global_pulse_delta_1 = self._parameters["global_pulse_delta_1"]
        global_pulse_delta_2 = self._parameters["global_pulse_delta_2"]
        # global_pulse_delta_3 = self._parameters["global_pulse_delta_3"]
        # global_pulse_delta_4 = self._parameters["global_pulse_delta_4"]
        local_pulses_omega_1 = torch.abs(self._parameters["local_pulses_omega_1"])
        # local_pulses_omega_2 = torch.abs(self._parameters["local_pulses_omega_2"])
        global_pulse_delta_1 = self._parameters["global_pulse_delta_1"]
        # global_pulse_delta_2 = self._parameters["global_pulse_delta_2"]
        local_pulses_delta_1 = self._parameters["local_pulses_delta_1"]
        # local_pulses_delta_2 = self._parameters["local_pulses_delta_2"]
        global_pulse_duration = self._parameters["global_pulse_duration"]
        local_pulse_duration = self._parameters["local_pulse_duration"]
        embed_pulse_duration = self._parameters["embed_pulse_duration"]
        kernel_size = self.hparams["kernel_size"]
        n_qubits = self.hparams["n_qubits"]
        protocol = self.hparams["protocol"]

        reg = Register({"q" + str(i): pos for i, pos in enumerate(positions)})
        seq = Sequence(reg, MockDevice)
        x = seq.declare_variable("x", dtype=float, size=n_qubits)

        seq.declare_channel("rydberg_global", "rydberg_global")
        for i in range(n_qubits):
            seq.declare_channel(f"rydberg_local_q{i}", "rydberg_local")
            seq.target(f"q{i}", channel=f"rydberg_local_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega_1 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta_1 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # embedding pulses
        # x is already in the range [0, 1] due to normalization
        for i in range(n_qubits):
            pulse_local = Pulse.ConstantPulse(
                embed_pulse_duration,
                1000 * x[i] * np.pi / embed_pulse_duration,
                0.0,
                0.0,  # pyright: ignore
            )  # Use x[i] as the amplitude
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)

        # local pulses (including ancilliary qubits)
        for i in range(n_qubits):
            pulse_local = Pulse.ConstantPulse(
                local_pulse_duration,
                local_pulses_omega_1[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                local_pulses_delta_1[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                0.0,
            )
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)
            # seq.declare_variable(f"omega_q{i}")
            # seq.declare_variable(f"delta_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega_2 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta_2 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # # embedding pulses (data reuploading)
        # for i in range(n_features):
        #     pulse_local = Pulse.ConstantPulse(
        #         embed_pulse_duration,
        #         1000 * x[i] * np.pi / embed_pulse_duration,
        #         0.0,
        #         0.0,  # pyright: ignore
        #     )  # Use x[i] as the amplitude
        #     seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)
        #     # seq.declare_variable(f"omega2_q{i}")
        #     # seq.declare_variable(f"delta2_q{i}")
        #
        # # global pulse
        # pulse_global = Pulse.ConstantPulse(
        #     global_pulse_duration,
        #     global_pulse_omega_3 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
        #     global_pulse_delta_3 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
        #     0.0,
        # )
        # seq.add(pulse_global, "rydberg_global")
        #
        # # local pulses (including ancilliary qubits)
        # for i in range(n_qubits):
        #     pulse_local = Pulse.ConstantPulse(
        #         local_pulse_duration,
        #         local_pulses_omega_2[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
        #         local_pulses_delta_2[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
        #         0.0,
        #     )
        #     seq.add(pulse_local, f"rydberg_local_q{i}", protocol="min-delay")
        #     # seq.declare_variable(f"omega3_q{i}")
        #     # seq.declare_variable(f"delta3_q{i}")
        #
        # # global pulse
        # pulse_global = Pulse.ConstantPulse(
        #     global_pulse_duration,
        #     global_pulse_omega_4 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
        #     global_pulse_delta_4 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
        #     0.0,
        # )
        # seq.add(pulse_global, "rydberg_global")

        return seq

    def forward(
        self,
        x: Tensor,
        time_grad: bool = False,
        dist_grad: bool = False,
        solver: str = "DP5_SE",
    ) -> dict:
        """
        parameters
        - x: Tensor
        - time_grad: bool, whether to store the gradients for all evaluation times, allowing derivatives w/r to these times
        - dist_grad: bool, allowes calculation for derivatives w/r to the inter-qubit distances r_ij
        - solver: SolverType, the solver to use for the simulation

        """
        if solver == "DP5_SE":
            solver = SolverType.DP5_SE
        elif solver == "KRYLOV_SE":
            solver = SolverType.KRYLOV_SE
        out = {"results": [], "output": []}
        seq_len = x.shape[0]
        sampling_rate = self.hparams["sampling_rate"]
        n_qubits = self.hparams["n_qubits"]
        kernel_size = self.hparams["kernel_size"]
        stride = self.hparams.get("stride", 3)

        for pos in range(0, seq_len - kernel_size + 1, stride):
            x_slice = x[pos : pos + kernel_size]
            base_seq = self.setup_register()
            x_slice = self.embedding_FC(x_slice)
            seq_built = base_seq.build(x=x_slice)
            sim = TorchEmulator.from_sequence(seq_built, sampling_rate=sampling_rate)
            if self.training:
                results = sim.run(
                    time_grad=time_grad, dist_grad=dist_grad, solver=SolverType.DP5_SE
                )
            else:
                with torch.no_grad():
                    results = sim.run(
                        time_grad=False, dist_grad=False, solver=SolverType.DP5_SE
                    )
            output = state_to_output_learned(
                results.states[-1], self.classical_params["conv_params"]
            ).squeeze()
            out["output"].append(output)
            out["results"].append(results)
        out["output"] = torch.stack(out["output"])
        out["output"] = self.fc_final(out["output"])

        return out


class CNN_1D(nn.Module):
    """
    Corresponding classical CNN for comparison
    """

    def __init__(self, hparams: dict):
        super().__init__()
        self.seq_len = hparams["input_length"]
        self.kenel_size = hparams["kernel_size"]
        self.stride = hparams["stride"]
        self.conv1_channels = hparams.get("conv1_channels", 1)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.conv1_channels,
            kernel_size=self.kenel_size,
            stride=self.stride,
        )
        self.dropout_val = hparams["dropout"]
        sample_input = torch.randn(
            1, 1, self.seq_len
        )  # batch_size=1, channels=1, seq_len
        self.conv1_out_len = self.conv1(sample_input).shape[-1]

        # set up final FC NN
        hidden_layers_dims = hparams.get("hidden_layers_dims", [])
        self.fc_final = nn.Sequential()
        # output length of the convolution
        current_dim = self.conv1_out_len
        for dim in hidden_layers_dims:
            self.fc_final.append(nn.Linear(current_dim, dim, dtype=torch.float32))
            self.fc_final.append(nn.Dropout(self.dropout_val))
            self.fc_final.append(nn.ReLU())
            current_dim = dim
        self.output_dim = hparams["output_dim"]
        self.fc_final.append(
            nn.Linear(current_dim, self.output_dim, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        input shape [N, Cin, Lin]
        """
        x = self.conv1(x)  # shape (batch_size, (channels) 1, conv_out_len)
        # pool over channels dimension
        x = x.mean(dim=1)  # shape (batch_size, conv_out_len)
        x = self.fc_final(x)  # shape (batch_size, output_dim)
        return x


class CNN_1D_Learned_Channel_Collapse(nn.Module):
    """
    Corresponding classical CNN for comparison
    """

    def __init__(self, hparams: dict):
        super().__init__()
        self.seq_len = hparams["input_length"]
        self.kenel_size = hparams["kernel_size"]
        self.stride = hparams["stride"]
        self.conv1_channels = hparams.get("conv1_channels", 1)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.conv1_channels,
            kernel_size=self.kenel_size,
            stride=self.stride,
        )
        self.dropout_val = hparams["dropout"]
        sample_input = torch.randn(
            1, 1, self.seq_len
        )  # batch_size=1, channels=1, seq_len
        self.conv1_out_len = self.conv1(sample_input).shape[-1]

        # set up final FC NN
        hidden_layers_dims = hparams.get("hidden_layers_dims", [])
        self.fc_final = nn.Sequential()
        # output length of the convolution
        current_dim = self.conv1_out_len
        self.fc_channel_collapse = nn.Linear(
            self.conv1_channels, 1, dtype=torch.float32
        )
        for dim in hidden_layers_dims:
            self.fc_final.append(nn.Linear(current_dim, dim, dtype=torch.float32))
            self.fc_final.append(nn.Dropout(self.dropout_val))
            self.fc_final.append(nn.ReLU())
            current_dim = dim
        self.output_dim = hparams["output_dim"]
        self.fc_final.append(
            nn.Linear(current_dim, self.output_dim, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        input shape [N, Cin, Lin]
        """
        x = self.conv1(x)  # shape (batch_size, conv1_channels, conv_out_len)
        # collapse channels dimension
        x = self.fc_channel_collapse(x.transpose(1, 2)).squeeze(
            2
        )  # shape (batch_size, conv_out_len)
        x = self.fc_final(x)  # shape (batch_size, output_dim)
        return x


def test_NAHEA_CNN_2():
    seq_len = 15
    hparams = {
        "n_qubits": (n_qubits := 5),
        "kernel_size": (n_features := 11),  # =1D kernel size = number of qubits
        "sampling_rate": 0.2,
        "protocol": "min-delay",
        "n_ancilliary_qubits": (n_ancilliary_qubits := 0),  # not implemented
        "input_length": seq_len,
        "stride": 4,  # stride for the convolution
        "output_dim": 2,  # output dimension of the final FC NN
        "hidden_layers_dims": [10, 5],
        "embedding_FC_hidden_dims": [8],
    }

    sep = 7.0
    parameters = {
        # separation of 7 between the qubits
        "positions": [[sep * i - (sep * 2), 0] for i in range(n_qubits)],
        # "local_pulses_omega_1": [0.5, 1.0, 1.5, 1.0, 0.5, 1.0, 1.5],
        "local_pulses_omega_1": [
            1.0 + np.sin(i * np.pi / 6) / 5 for i in range(n_qubits)
        ],
        "local_pulses_delta_1": [0.0] * n_qubits,
        "global_pulse_omega_1": 1.0,
        "global_pulse_delta_1": 0.0,
        "global_pulse_omega_2": 0.5,
        "global_pulse_delta_2": 0.0,
        "global_pulse_duration": 100,
        "local_pulse_duration": 80,
        "embed_pulse_duration": 80,
    }

    NAHEA_CNN = NAHEA_CNN_2(
        hparams=hparams, parameters=parameters, name="test_model_2features"
    )

    # # generate some training data
    n_samples = 220

    ## generate sin wave data with different frequencies
    def generate_sin_wave_data(n_samples, seq_len):
        X = []
        y = []
        for i in range(n_samples):
            freq = np.random.uniform(0.1, 1.0)  # frequency between 0.1 and 1.0
            x = np.linspace(0, 2 * np.pi, seq_len)
            seq = np.sin(freq * x)
            X.append(seq)
            y.append(freq)  # target is the frequency
        X = np.array(X).reshape(n_samples, seq_len, 1)  # shape (n_samples, seq_len, 1)
        y = np.array(y).reshape(n_samples, 1)  # shape (n_samples, 1)
        return X, y

    X, y_true = generate_sin_wave_data(n_samples, seq_len)
    # normalize X to [0, 1]
    X = (X - X.min()) / (X.max() - X.min())
    print(f"Generated data shapes: {X.shape}, {y_true.shape}")

    # train test split
    percentage = 0.8
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=1 - percentage, random_state=42
    )


if __name__ == "__main__":
    seq_len = 15
    hparams = {
        "n_features": (n_features := 4),
        "sampling_rate": 0.2,
        "protocol": "min-delay",
        "n_ancilliary_qubits": (n_ancilliary_qubits := 0),
        "input_length": seq_len,
        "stride": 3,  # stride for the convolution
        "output_dim": 1,  # output dimension of the final FC NN
        "hidden_layers_dims": [10, 5],
    }
    sep = 7
    parameters = {
        # separation of 7 between the qubits
        "positions": [[sep * i - (sep * 2), 0] for i in range(n_features)],
        "local_pulses_omega_1": [0.5, 1.5, 1.5, 0.5],
        "local_pulses_delta_1": [0.0] * n_features,
        "global_pulse_omega_1": 1.0,
        "global_pulse_delta_1": 0.0,
        "global_pulse_omega_2": 0.5,
        "global_pulse_delta_2": 0.0,
        "global_pulse_duration": 100,
        "local_pulse_duration": 80,
        "embed_pulse_duration": 80,
    }

    NAHEA_CNN = NAHEA_CNN_1(
        hparams=hparams, parameters=parameters, name="test_model_2features"
    )

    # # generate some training data
    n_samples = 220

    ## generate sin wave data with different frequencies
    def generate_sin_wave_data(n_samples, seq_len):
        X = []
        y = []
        for i in range(n_samples):
            freq = np.random.uniform(0.1, 1.0)  # frequency between 0.1 and 1.0
            x = np.linspace(0, 2 * np.pi, seq_len)
            seq = np.sin(freq * x)
            X.append(seq)
            y.append(freq)  # target is the frequency
        X = np.array(X).reshape(n_samples, seq_len, 1)  # shape (n_samples, seq_len, 1)
        y = np.array(y).reshape(n_samples, 1)  # shape (n_samples, 1)
        return X, y

    X, y_true = generate_sin_wave_data(n_samples, seq_len)
    # normalize X to [0, 1]
    X = (X - X.min()) / (X.max() - X.min())
    print(f"Generated data shapes: {X.shape}, {y_true.shape}")

    # train test split
    percentage = 0.8
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=1 - percentage, random_state=42
    )

    hparams_classical = {
        "kernel_size": n_features,
        "input_length": seq_len,
        "stride": 3,  # stride for the convolution
        "output_dim": 1,  # output dimension of the final FC NN
        "hidden_layers_dims": [10, 5],
    }
    classical_CNN = CNN_1D(hparams_classical)
    i = 2
    x = torch.tensor(X[i : i + 1], dtype=torch.float32).transpose(1, 2)
    print(f"{x.shape=}")
    out = classical_CNN(x)
    print(f"{out=}")

    # # use NAHEA_kernel like a kernel in a CNN
    # NAHEA_CNN.eval()
    # i = 2
    # out = NAHEA_CNN(X[i])
    # print(f"{out=}")
    #
    # # plot x and at the corresponding point, the output of the NAHEA_CNN
    # # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # # plot_x = np.linspace(0, 2 * np.pi, seq_len)
    # # plot_y = X[i].squeeze()
    # # plot_y_out = out["output"].detach().numpy()
    # # stride = 3
    # # out_x_points = np.array(range(0, seq_len - n_features + 1, stride))
    # # plt.plot(plot_x, plot_y, label="Input Signal")
    # # plt.scatter(plot_x[out_x_points], plot_y_out, color="red", label="NAHEA Output")
    # # plt.show()
    #
    # # train the model
    # n_train = 64
    # X_train = X_train[:n_train]
    # batch_size = 16
    # epochs = 5
    # loss_hist = []
    # # MSE loss function
    # loss_fn = torch.nn.MSELoss()
    # NAHEA_CNN.train()
    # print(f"{NAHEA_CNN.parameters()}")
    # optimizer = torch.optim.Adam(
    #     [param for _, param in NAHEA_CNN.parameters().items() if param.requires_grad],
    # )
    # params_hist = [copy.deepcopy(NAHEA_CNN.parameters())]
    # for epoch in range(epochs):
    #     # shuffle X_train and y_train
    #     idxs = np.arange(len(X_train))
    #     idxs = np.random.permutation(idxs)
    #     X_train = X_train[idxs]
    #
    #     epoch_losses = []
    #     for i in tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}"):
    #         optimizer.zero_grad()
    #         x_batch = torch.tensor(X_train[i : i + batch_size], dtype=torch.float64)
    #         y_batch = torch.tensor(
    #             y_train[i : i + batch_size], dtype=torch.float64
    #         ).squeeze(1)
    #         batch_out = [NAHEA_CNN(xx) for xx in x_batch]
    #         predicted = torch.stack([bo["output"] for bo in batch_out])
    #         loss = loss_fn(predicted.squeeze(1), y_batch)
    #         tqdm.write(
    #             f"Batch {i // batch_size + 1}/{n_samples // batch_size}, Loss: {loss.item()}"
    #         )
    #         epoch_losses.append(loss.item())
    #         loss_hist.append(loss.item())
    #         loss.backward()
    #         optimizer.step()
    #         params_hist.append(copy.deepcopy(NAHEA_CNN.parameters()))
    #     epoch_loss = np.mean(epoch_losses)
    #     tqdm.write(f"Epoch {epoch+1} train loss: {epoch_loss}")
    # loss_hist = np.array(loss_hist)
    # print(f"Final loss: {loss_hist[-1]}")
    #
    # # test the model
    # NAHEA_CNN.eval()
    # y_pred_test = []
    # for i in range(len(X_test)):
    #     x_test = torch.tensor(X_test[i], dtype=torch.float64)
    #     pred = NAHEA_CNN(x_test)["output"].item()
    #     y_pred_test.append(pred)
    # y_pred_test = torch.tensor(y_pred_test, dtype=torch.float64)
    # y_test = torch.tensor(y_test, dtype=torch.float64).squeeze(1)
    # loss_test = loss_fn(y_pred_test, y_test)
    # print(f"Test loss: {loss_test.item()}")
    # print("final parameters:")
    # print(NAHEA_CNN.parameters())
    #
    # plt.plot(loss_hist, label="Training Loss")
    # plt.xlabel("Batch")
    # plt.ylabel("Loss")
    # plt.title("Training Loss History")
    # plt.legend()
    # plt.show()
