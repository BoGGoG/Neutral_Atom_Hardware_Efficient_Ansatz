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

    def parameters(self):
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

    def __call__(self, x):
        return self.forward(x)

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


class NAHEA_nFeatures_1(NAHEA):
    """NAHEA model with n features."""

    def __init__(
        self, hparams: dict, parameters: dict, name: str = "NAHEA_nFeatures_1 model"
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
        # seq.declare_variable("omega_global")
        # seq.declare_variable("delta_global")

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
        # seq.declare_variable("omega1_global")
        # seq.declare_variable("delta1_global")

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
        # seq.declare_variable("omega2_global")
        # seq.declare_variable("delta2_global")

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
        # seq.declare_variable("omega3_global")
        # seq.declare_variable("delta3_global")

        return seq


if __name__ == "__main__":
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

    model = NAHEA_nFeatures_1(
        hparams=hparams, parameters=parameters, name="test_model_2features"
    )

    seq = model.setup_register()
    print("Sequence created successfully.")
    print(seq)
    print(f"{seq.is_parametrized()=}")

    x = torch.tensor([0.5, 0.5], dtype=torch.float32)
    seq_built = seq.build(x=x)
    print(f"{seq.is_parametrized()=}")
    # seq_built.draw()

    sampling_rate = hparams["sampling_rate"]
    sim = TorchEmulator.from_sequence(seq_built, sampling_rate=sampling_rate)
    results = sim.run(time_grad=True, dist_grad=True, solver=SolverType.DP5_SE)

    output = state_to_output(results.states[-1])
    print(f"{output=}")
    loss = 1 - output
    loss.backward()

    # get gradients
    print(model.parameters())
    print(model._parameters)
    pos_tensor = model.parameters()["positions"]
    print(f"{pos_tensor.grad=}")
