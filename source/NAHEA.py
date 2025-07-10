import copy
import os
import sys
from pathlib import Path

import numpy as np
from qutip.solver.floquet import progress_bars
import torch
from pulser import Pulse, Register, Sequence
from pulser.devices import MockDevice
from pulser_diff.backend import TorchEmulator
from pyqtorch.utils import SolverType
from torch import Tensor
import json
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


class NAHEA:
    """Neutral Atoms Hardware Efficient Ansatz for Deep Learning (NAHEA)
    This is a base class for NAHEA models.
    Can't use PyTorch here, so I need to immitate a pytorch-like interface.
    """

    def __init__(
        self,
        hparams: dict,
        parameters: dict,
        name: str = "NAHEA model",
    ):
        self.name = name
        self.training: bool = False
        self.hparams_required_keys = getattr(self, "hparams_required_keys", [])
        self.parameters_required_keys = getattr(self, "parameters_required_keys", [])
        self.non_trainable_params = getattr(self, "non_trainable_params", [])
        self.hparams: dict = copy.deepcopy(hparams)
        self.check_hparams()
        self._parameters: dict = copy.deepcopy(parameters)
        self.check_parameters()
        self.params_to_tensors()

    def __str__(self):
        return f"NAHEA(name={self.name})"

    def forward(self, x):
        """Forward pass of the model."""
        # Placeholder for forward pass logic
        return x

    def parameters(self) -> dict:
        """Return an iterator over model parameters."""
        return self._parameters

    def save(self):
        """Save model state to a file."""
        pass

    def load(self):
        """Load model state from a file."""
        pass

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        self.training = mode

    def eval(self):
        """Set the model to evaluation mode."""
        self.train(False)

    def __call__(self, x, **args):
        return self.forward(x, *args)

    def check_hparams(self):
        """Check hyperparameters."""
        if not isinstance(self.hparams, dict):
            raise ValueError("Hyperparameters must be a dictionary.")
        # Add more checks as needed
        keys = self.hparams.keys()

        # check that hparams has the required keys
        required_keys = self.hparams_required_keys
        absent_keys = [key for key in required_keys if key not in keys]
        if absent_keys:
            raise ValueError(
                f"Missing required hyperparameters: {', '.join(absent_keys)}"
            )

    def check_parameters(self):
        """Check parameters."""
        if not isinstance(self._parameters, dict):
            raise ValueError("Parameters must be a dictionary.")
        # Add more checks as needed
        keys = self._parameters.keys()

        required_keys = self.parameters_required_keys
        # check that parameters has the required keys
        absent_keys = [key for key in required_keys if key not in keys]
        if absent_keys:
            raise ValueError(f"Missing required parameters: {', '.join(absent_keys)}")

    def params_to_tensors(self):
        """convert parameters to torch tensors."""
        non_trainable_params = self.non_trainable_params
        for key, value in self._parameters.items():
            if key not in non_trainable_params:
                self._parameters[key] = torch.tensor(
                    value, dtype=torch.float32, requires_grad=True
                )
            else:
                self._parameters[key] = torch.tensor(
                    value, dtype=torch.float32, requires_grad=False
                )

    def state_dict(self) -> dict:
        """Return the state dictionary of the model."""
        return {
            "name": self.name,
            "training": self.training,
            "hparams": self.hparams,
            "parameters": self._parameters,
        }

    def save_state_dict(self, filepath: str, verbose: bool = True):
        """Save the state dictionary to a file using JSON format."""

        def tensor_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [tensor_to_serializable(v) for v in obj]
            else:
                return obj

        state = self.state_dict()
        state = tensor_to_serializable(state)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=4)
        if verbose:
            print(f"Model state saved to {filepath}")

    def load_state_dict_from_json(self, filepath: str):
        """Load the state dictionary from a JSON file and convert lists back to tensors."""

        def serializable_to_tensor(obj):
            if (
                isinstance(obj, list)
                or isinstance(obj, np.ndarray)
                or isinstance(obj, float)
            ):
                # Recursively convert nested lists to tensors
                return torch.tensor(obj)
            elif isinstance(obj, dict):
                return {k: serializable_to_tensor(v) for k, v in obj.items()}
            else:
                return obj

        with open(filepath, "r") as f:
            state = json.load(f)
        state = serializable_to_tensor(state)
        self.load_state_dict(state)

    def load_state_dict(self, state_dict: dict):
        self.name = state_dict.get("name", self.name)
        self.training = state_dict.get("training", self.training)
        self.hparams = state_dict.get("hparams", self.hparams)
        self._parameters = state_dict.get("parameters", self._parameters)
        self.params_to_tensors()

    def load_from_file(self, filepath: str):
        self.load_state_dict(self._load_json(filepath))

    def _load_json(self, filepath: str) -> dict:
        with open(filepath, "r") as f:
            obj = json.load(f)
        return obj

    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath, "r") as f:
            state = json.load(f)
        hparams = state.get("hparams", {})
        parameters = state.get("parameters", {})
        name = state.get("name", "NAHEA model")
        model = cls(hparams, parameters, name)
        model.load_state_dict(state)
        return model


class NAHEA_nFeatures_BinClass_1(NAHEA):
    """NAHEA model with n features."""

    def __init__(
        self,
        hparams: dict,
        parameters: dict,
        name: str = "NAHEA_nFeatures_BinClass_1 model",
    ):
        """ """
        self.parameters_required_keys = [
            "positions",
            "local_pulses_omega",
            "local_pulses_delta",
            "global_pulse_omega",
            "global_pulse_delta",
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
        ]
        super().__init__(hparams, parameters, name)
        self.input_checks()

    def input_checks(self):
        assert (
            len(self._parameters["positions"])
            == len(self._parameters["local_pulses_omega"])
            == len(self._parameters["local_pulses_delta"])
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
        global_pulse_omega = torch.abs(self._parameters["global_pulse_omega"])
        local_pulses_omega = torch.abs(self._parameters["local_pulses_omega"])
        global_pulse_delta = self._parameters["global_pulse_delta"]
        global_pulse_duration = self._parameters["global_pulse_duration"]
        local_pulse_duration = self._parameters["local_pulse_duration"]
        local_pulses_delta = self._parameters["local_pulses_delta"]
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

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # local pulses (including ancilliary qubits)
        for i in range(n_qubits):
            pulse_local = Pulse.ConstantPulse(
                local_pulse_duration,
                local_pulses_omega[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                local_pulses_delta[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                0.0,
            )
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)
            # seq.declare_variable(f"omega_q{i}")
            # seq.declare_variable(f"delta_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # embedding pulses (data reuploading)
        for i in range(n_features):
            pulse_local = Pulse.ConstantPulse(
                embed_pulse_duration,
                1000 * x[i] * np.pi / embed_pulse_duration,
                0.0,
                0.0,  # pyright: ignore
            )  # Use x[i] as the amplitude
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)
            # seq.declare_variable(f"omega2_q{i}")
            # seq.declare_variable(f"delta2_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # local pulses (including ancilliary qubits)
        for i in range(n_qubits):
            pulse_local = Pulse.ConstantPulse(
                local_pulse_duration,
                local_pulses_omega[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                local_pulses_delta[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                0.0,
            )
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol="min-delay")
            # seq.declare_variable(f"omega3_q{i}")
            # seq.declare_variable(f"delta3_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
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

        base_seq = self.setup_register()
        seq_built = base_seq.build(x=x)
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
        output = state_to_output(results.states[-1]).squeeze()
        out = {
            "sim_evaluation_times": sim.evaluation_times,
            "results": results,
            "output": output,
        }

        return out


class NAHEA_nFeatures_BinClass_2(NAHEA):
    """NAHEA model with n features."""

    def __init__(
        self,
        hparams: dict,
        parameters: dict,
        name: str = "NAHEA_nFeatures_BinClass_2 model",
    ):
        """ """
        self.parameters_required_keys = [
            "positions",
            "local_pulses_omega",
            "local_pulses_delta",
            "global_pulse_omega",
            "global_pulse_delta",
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
        ]
        super().__init__(hparams, parameters, name)
        self.input_checks()

    def input_checks(self):
        assert (
            len(self._parameters["positions"])
            == len(self._parameters["local_pulses_omega"])
            == len(self._parameters["local_pulses_delta"])
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
        global_pulse_omega = torch.abs(self._parameters["global_pulse_omega"])
        local_pulses_omega = torch.abs(self._parameters["local_pulses_omega"])
        global_pulse_delta = self._parameters["global_pulse_delta"]
        global_pulse_duration = self._parameters["global_pulse_duration"]
        local_pulse_duration = self._parameters["local_pulse_duration"]
        local_pulses_delta = self._parameters["local_pulses_delta"]
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
            global_pulse_omega * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
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

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # local pulses (including ancilliary qubits)
        for i in range(n_qubits):
            pulse_local = Pulse.ConstantPulse(
                local_pulse_duration,
                local_pulses_omega[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                local_pulses_delta[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                0.0,
            )
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)
            # seq.declare_variable(f"omega_q{i}")
            # seq.declare_variable(f"delta_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # embedding pulses (data reuploading)
        for i in range(n_features):
            pulse_local = Pulse.ConstantPulse(
                embed_pulse_duration,
                1000 * x[i] * np.pi / embed_pulse_duration,
                0.0,
                0.0,  # pyright: ignore
            )  # Use x[i] as the amplitude
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)
            # seq.declare_variable(f"omega2_q{i}")
            # seq.declare_variable(f"delta2_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # local pulses (including ancilliary qubits)
        for i in range(n_qubits):
            pulse_local = Pulse.ConstantPulse(
                local_pulse_duration,
                local_pulses_omega[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                local_pulses_delta[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                0.0,
            )
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol="min-delay")
            # seq.declare_variable(f"omega3_q{i}")
            # seq.declare_variable(f"delta3_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
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

        base_seq = self.setup_register()
        seq_built = base_seq.build(x=x)
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
        output = state_to_output(results.states[-1]).squeeze()
        out = {
            "sim_evaluation_times": sim.evaluation_times,
            "results": results,
            "output": output,
        }

        return out


class NAHEA_nFeatures_BinClass_3(NAHEA):
    """NAHEA model with n features.
    Now with more parameters
    """

    def __init__(
        self,
        hparams: dict,
        parameters: dict,
        name: str = "NAHEA_nFeatures_BinClass_3 model",
    ):
        """ """
        self.parameters_required_keys = [
            "positions",
            "local_pulses_omega_1",
            "local_pulses_omega_2",
            "local_pulses_delta_1",
            "local_pulses_delta_2",
            "global_pulse_omega_1",
            "global_pulse_omega_2",
            "global_pulse_omega_3",
            "global_pulse_omega_4",
            "global_pulse_delta_1",
            "global_pulse_delta_2",
            "global_pulse_delta_3",
            "global_pulse_delta_4",
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
        ]
        super().__init__(hparams, parameters, name)
        self.input_checks()

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
        global_pulse_omega_3 = torch.abs(self._parameters["global_pulse_omega_3"])
        global_pulse_omega_4 = torch.abs(self._parameters["global_pulse_omega_4"])
        global_pulse_delta_1 = self._parameters["global_pulse_delta_1"]
        global_pulse_delta_2 = self._parameters["global_pulse_delta_2"]
        global_pulse_delta_3 = self._parameters["global_pulse_delta_3"]
        global_pulse_delta_4 = self._parameters["global_pulse_delta_4"]
        local_pulses_omega_1 = torch.abs(self._parameters["local_pulses_omega_1"])
        local_pulses_omega_2 = torch.abs(self._parameters["local_pulses_omega_2"])
        global_pulse_delta_1 = self._parameters["global_pulse_delta_1"]
        global_pulse_delta_2 = self._parameters["global_pulse_delta_2"]
        local_pulses_delta_1 = self._parameters["local_pulses_delta_1"]
        local_pulses_delta_2 = self._parameters["local_pulses_delta_2"]
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

        # embedding pulses (data reuploading)
        for i in range(n_features):
            pulse_local = Pulse.ConstantPulse(
                embed_pulse_duration,
                1000 * x[i] * np.pi / embed_pulse_duration,
                0.0,
                0.0,  # pyright: ignore
            )  # Use x[i] as the amplitude
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)
            # seq.declare_variable(f"omega2_q{i}")
            # seq.declare_variable(f"delta2_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega_3 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta_3 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # local pulses (including ancilliary qubits)
        for i in range(n_qubits):
            pulse_local = Pulse.ConstantPulse(
                local_pulse_duration,
                local_pulses_omega_2[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                local_pulses_delta_2[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                0.0,
            )
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol="min-delay")
            # seq.declare_variable(f"omega3_q{i}")
            # seq.declare_variable(f"delta3_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega_4 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta_4 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
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

        base_seq = self.setup_register()
        seq_built = base_seq.build(x=x)
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
        output = state_to_output(results.states[-1]).squeeze()
        out = {
            "sim_evaluation_times": sim.evaluation_times,
            "results": results,
            "output": output,
        }

        return out


class NAHEA_nFeatures_BinClass_4(NAHEA):
    """NAHEA model with n features.
    Now with a learned output layer
    """

    def __init__(
        self,
        hparams: dict,
        parameters: dict,
        name: str = "NAHEA_nFeatures_BinClass_4 model",
    ):
        """ """
        self.parameters_required_keys = [
            "positions",
            "local_pulses_omega_1",
            "local_pulses_omega_2",
            "local_pulses_delta_1",
            "local_pulses_delta_2",
            "global_pulse_omega_1",
            "global_pulse_omega_2",
            "global_pulse_omega_3",
            "global_pulse_omega_4",
            "global_pulse_delta_1",
            "global_pulse_delta_2",
            "global_pulse_delta_3",
            "global_pulse_delta_4",
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
        super().__init__(hparams, parameters, name)

        num_states = 2 ** len(parameters["positions"])  # number of possible states
        hidden_layers_dims = hparams.get("hidden_layers_dims", [])
        self.fc_final = nn.Sequential()
        # output length of the convolution
        current_dim = num_states
        for dim in hidden_layers_dims:
            self.fc_final.append(nn.Linear(current_dim, dim, dtype=torch.float64))
            self.fc_final.append(nn.ReLU())
            current_dim = dim
        self.output_dim = hparams["output_dim"]
        self.fc_final.append(
            nn.Linear(current_dim, self.output_dim, dtype=torch.float64)
        )
        print(self.fc_final)

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

        self.input_checks()

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
        global_pulse_omega_3 = torch.abs(self._parameters["global_pulse_omega_3"])
        global_pulse_omega_4 = torch.abs(self._parameters["global_pulse_omega_4"])
        global_pulse_delta_1 = self._parameters["global_pulse_delta_1"]
        global_pulse_delta_2 = self._parameters["global_pulse_delta_2"]
        global_pulse_delta_3 = self._parameters["global_pulse_delta_3"]
        global_pulse_delta_4 = self._parameters["global_pulse_delta_4"]
        local_pulses_omega_1 = torch.abs(self._parameters["local_pulses_omega_1"])
        local_pulses_omega_2 = torch.abs(self._parameters["local_pulses_omega_2"])
        global_pulse_delta_1 = self._parameters["global_pulse_delta_1"]
        global_pulse_delta_2 = self._parameters["global_pulse_delta_2"]
        local_pulses_delta_1 = self._parameters["local_pulses_delta_1"]
        local_pulses_delta_2 = self._parameters["local_pulses_delta_2"]
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

        # embedding pulses (data reuploading)
        for i in range(n_features):
            pulse_local = Pulse.ConstantPulse(
                embed_pulse_duration,
                1000 * x[i] * np.pi / embed_pulse_duration,
                0.0,
                0.0,  # pyright: ignore
            )  # Use x[i] as the amplitude
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol=protocol)
            # seq.declare_variable(f"omega2_q{i}")
            # seq.declare_variable(f"delta2_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega_3 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta_3 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            0.0,
        )
        seq.add(pulse_global, "rydberg_global")

        # local pulses (including ancilliary qubits)
        for i in range(n_qubits):
            pulse_local = Pulse.ConstantPulse(
                local_pulse_duration,
                local_pulses_omega_2[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                local_pulses_delta_2[i] * np.pi * 1000 / local_pulse_duration,  # pyright: ignore
                0.0,
            )
            seq.add(pulse_local, f"rydberg_local_q{i}", protocol="min-delay")
            # seq.declare_variable(f"omega3_q{i}")
            # seq.declare_variable(f"delta3_q{i}")

        # global pulse
        pulse_global = Pulse.ConstantPulse(
            global_pulse_duration,
            global_pulse_omega_4 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
            global_pulse_delta_4 * np.pi * 1000 / global_pulse_duration,  # pyright: ignore
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
        only takes a single element, no batch support yet
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

        base_seq = self.setup_register()
        seq_built = base_seq.build(x=x)
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
        states = results.states[-1]
        output = self.fc_final(states.abs().view(-1))
        output = torch.sigmoid(output).squeeze()  # apply sigmoid to the output
        out = {
            "sim_evaluation_times": sim.evaluation_times,
            "results": results,
            "output": output,
        }

        return out


def test_NAHEA_nFeatures_BinClass_2():
    hparams = {
        "n_features": 2,
        "sampling_rate": 0.4,
        "protocol": "min-delay",
        "n_ancilliary_qubits": (n_ancilliary_qubits := 0),
    }
    parameters = {
        "positions": [[-3.6672354, 0.0], [3.6672359, 0.0]],
        "local_pulses_omega": [1.1559689, 1.6583259],
        "local_pulses_delta": [-0.76122487, 1.5434982],
        "global_pulse_omega": -0.26719406,
        "global_pulse_delta": 1.0807998,
        "global_pulse_duration": 50,
        "local_pulse_duration": 50,
        "embed_pulse_duration": 80,
    }

    model = NAHEA_nFeatures_BinClass_1(
        hparams=hparams, parameters=parameters, name="test_model_2features"
    )

    # seq = model.setup_register()
    # print("Sequence created successfully.")
    # print(seq)
    # print(f"{seq.is_parametrized()=}")
    #
    x = torch.tensor([0.5, 0.5], dtype=torch.float32)
    # seq_built = seq.build(x=x)
    # print(f"{seq.is_parametrized()=}")
    # # seq_built.draw()
    #
    # sampling_rate = hparams["sampling_rate"]
    # sim = TorchEmulator.from_sequence(seq_built, sampling_rate=sampling_rate)
    # results = sim.run(time_grad=False, dist_grad=True, solver=SolverType.DP5_SE)
    model.train()  # set model to training mode
    out = model(
        x,
        time_grad=False,
        dist_grad=True,
        solver="DP5_SE",
    )  # use DP5_SE solver for now
    results = out["results"]
    print(f"result.states: {out['results'].states.shape}")

    output = state_to_output(results.states[-1])
    print(f"{output=}")
    loss = 1 - output
    loss.backward()

    # get gradients
    print(model.parameters())
    print(model._parameters)
    pos_tensor = model.parameters()["positions"]
    print(f"{pos_tensor.grad=}")

    # no gradients
    model.eval()  # set model to evaluation mode
    out = model(
        x,
        time_grad=False,
        dist_grad=True,
        solver="DP5_SE",
    )  # use DP5_SE solver for now

    model_save_path = Path("dev") / "models" / "test_model_2features.json"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # save model state
    model.save_state_dict(model_save_path)
    print(f"Model state saved to {model_save_path}")
    print("Model state dictionary:")
    # model.load_state_dict_from_json(model_save_path)
    # print(f"Model state loaded from {model_save_path}")

    loaded_model = model.from_file(model_save_path)

    print(model)
    print(loaded_model)


def test_NAHEA_nFeatures_BinClass_4():
    hparams = {
        "n_features": (n_features := 2),
        "sampling_rate": 0.4,
        "protocol": "min-delay",
        "n_ancilliary_qubits": (n_ancilliary_qubits := 0),
        "output_dim": 1,  # output dimension for the final layer
        "hidden_layers_dims": [],
    }
    sep = 6.8
    parameters = {
        # separation of 7 between the qubits
        "positions": [[sep * i - (sep * 2), 0] for i in range(n_features)],
        "local_pulses_omega_1": [0.5, 0.5],
        "local_pulses_delta_1": [0.0] * n_features,
        "local_pulses_omega_2": [0.5, 0.5],
        "local_pulses_delta_2": [0.0] * n_features,
        "global_pulse_omega_1": 1.0,
        "global_pulse_delta_1": 0.0,
        "global_pulse_omega_2": 1.0,
        "global_pulse_delta_2": 0.0,
        "global_pulse_omega_3": 1.0,
        "global_pulse_delta_3": 0.0,
        "global_pulse_omega_4": 1.0,
        "global_pulse_delta_4": 0.0,
        "global_pulse_duration": 100,
        "local_pulse_duration": 80,
        "embed_pulse_duration": 80,
    }

    model = NAHEA_nFeatures_BinClass_4(
        hparams=hparams,
        parameters=parameters,
        name="Test Model with learned output layer",
    )
    print(model)

    x = torch.tensor([0.5, 0.5], dtype=torch.float32)
    model.eval()  # set model to training mode
    out = model(
        x,
        time_grad=False,
        dist_grad=True,
        solver="DP5_SE",
    )  # use DP5_SE solver for now
    results = out["results"]
    print(f"result.states: {out['results'].states.shape}")
    print(f"{out['output']=}")
    print(f"{out['sim_evaluation_times'].shape=}")


if __name__ == "__main__":
    # test_NAHEA_nFeatures_BinClass_2()
    test_NAHEA_nFeatures_BinClass_4()
