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

    def __str__(self):
        return f"NAHEAD(name={self.name})"

    def forward(self, x):
        """Forward pass of the model."""
        # Placeholder for forward pass logic
        return x

    def parameters(self):
        """Return an iterator over model parameters."""
        return []

    def save(self, filepath):
        """Save model state to a file."""
        pass

    def load(self, filepath):
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


if __name__ == "__main__":
    print("asdf")

    model = NAHEA("test_model")
    print(model)
