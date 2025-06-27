import os
import sys
import torch
import copy


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


class NAHEA_2Features_1(NAHEA):
    """NAHEA model with 2 features."""

    def __init__(
        self, hparams: dict, parameters: dict, name: str = "NAHEA_2Features_1 model"
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
            "sampling_rate",
            "protocol",
            "n_ancilliary_qubits",
        ]
        super().__init__(hparams, parameters, name)
        self.input_checks()

    def input_checks(self):
        assert (
            len(self._parameters["positions"]) + self.hparams["n_ancilliary_qubits"]
            == len(self._parameters["local_pulses_omega"])
            == len(self._parameters["local_pulses_delta"])
        )
