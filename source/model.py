import os
import sys


class NAHEA:
    """Neutral Atoms Hardware Efficient Ansatz for Deep Learning (NAHEAD)
    This is a base class for NAHEA models.
    Can't use PyTorch here, so I need to immitate a pytorch-like interface.
    """

    def __init__(self, name: str = "NAHEA model"):
        self.name = name
        self.training: bool = False
        self._parameters: dict = {}

    def __str__(self):
        return f"NAHEAD(name={self.name})"

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


class NAHEA_2Features_1(NAHEA):
    """NAHEA model with 2 features."""

    def __init__(self, hparams: dict, name: str = "NAHEA_2Features_1 model"):
        super().__init__(name)
        self.hparams = hparams
        self.check_hparams()

    def check_hparams(self):
        """Check hyperparameters."""
        if not isinstance(self.hparams, dict):
            raise ValueError("Hyperparameters must be a dictionary.")
        # Add more checks as needed
        keys = self.hparams.keys()

        # check that hparams has the required keys
        required_keys = ["sampling_rate", "protocol", "n_ancilliary_qubits"]
        absent_keys = [key for key in required_keys if key not in keys]
        if absent_keys:
            raise ValueError(
                f"Missing required hyperparameters: {', '.join(absent_keys)}"
            )


if __name__ == "__main__":
    print("asdf")

    model = NAHEA("test_model")
    print(model)

    hparams = {
        "sampling_rate": 0.4,
        "protocol": "min-delay",
        "n_ancilliary_qubits": 0,
    }
    model1 = NAHEA_2Features_1(hparams)
    print(model1)
