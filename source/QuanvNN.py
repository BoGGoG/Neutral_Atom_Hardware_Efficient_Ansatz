import h5py as h5
from pathlib import Path

import math
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as pnp


def make_quantum_patch_torchlayer(
    patch_size: int,
    out_channels: int,
    depth: int = 2,
    dev_name: str = "default.qubit",
    shots: int = None,
):
    """
    Build a PennyLane TorchLayer that maps a length-K patch (K=patch_size)
    to out_channels features via expectation values.

    - Encoding: RX(angle) on each qubit with linear scaling of inputs to [0, pi].
    - Trainables: StronglyEntanglingLayers weights.
    - Measurements: Z expectation on first `out_channels` qubits.

    Returns: nn.Module compatible with Torch, input shape (..., patch_size),
             output shape (..., out_channels).
    """
    n_qubits = max(out_channels, patch_size)  # ensure we have >= out_channels
    dev = qml.device(dev_name, wires=n_qubits, shots=shots)

    # The QNode takes a 1D tensor of length patch_size as "inputs"
    @qml.qnode(dev, interface="torch", diff_method="best")
    def qnode(inputs, weights):
        # inputs: tensor shape (patch_size,)
        # Map patch to first patch_size qubits
        # Scale to [0, pi] for stability; add tiny epsilon to avoid NaNs if you later use inverse trig
        scaled = math.pi * inputs

        # Placeholders for missing qubits if patch_size < n_qubits
        x = torch.zeros(n_qubits, dtype=inputs.dtype, device=inputs.device)
        x[:patch_size] = scaled

        # Angle embedding
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="X")

        # Variational entangling template
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        # Measure Z on the first out_channels wires => features
        return [qml.expval(qml.PauliZ(i)) for i in range(out_channels)]

    # Weight spec for TorchLayer
    # StronglyEntanglingLayers expects shape (depth, n_wires, 3)
    weight_shapes = {"weights": (depth, n_qubits, 3)}

    # Wrap as a TorchLayer
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    return qlayer


class Quanv1D(nn.Module):
    """
    A 1D quanvolution layer:
      - Extracts sliding windows along time axis (like Conv1d kernel),
      - Feeds each window through a quantum patch TorchLayer,
      - Returns a feature map [B, C_out, L_out].

    Parameters
    ----------
    in_channels : int
        Must be 1 for this implementation (extend by channel mixing if needed).
    out_channels : int
        Number of quantum features per window.
    kernel_size : int
        Window length along time axis.
    stride : int
        Window stride.
    padding : int
        Zero-padding on both sides of the sequence.
    depth : int
        Entangling depth of the quantum circuit template.
    dev_name : str
        PennyLane device name.
    shots : Optional[int]
        None for analytic; integer for sampling (noisy gradient).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        depth: int = 2,
        dev_name: str = "default.qubit",
        shots: int = None,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                "This Quanv1D implementation expects in_channels=1. "
                "You can pre-mix channels with a 1x1 Conv1d before Quanv1D."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        # Build the quantum patch module
        self.qpatch = make_quantum_patch_torchlayer(
            patch_size=self.kernel_size,
            out_channels=self.out_channels,
            depth=depth,
            dev_name=dev_name,
            shots=shots,
        )

        # Simple learnable affine (optional): can help stabilize training
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, L]
        returns: [B, out_channels, L_out]
        """
        B, C, L = x.shape
        assert C == 1, "Use a 1x1 Conv1d to collapse to one channel first."

        # Use 2D unfold trick treating time as width (W) and height=1
        # Input for unfold must be [B, C, H, W]
        x4 = x.unsqueeze(2)  # [B,1,1,L]
        # Extract patches of size (1, kernel_size) with stride (1, stride)
        patches = F.unfold(
            x4,
            kernel_size=(1, self.kernel_size),
            dilation=(1, 1),
            padding=(0, self.padding),
            stride=(1, self.stride),
        )  # [B, C*1*kernel_size, L_out] = [B, kernel_size, L_out]
        print(f"{patches.shape=}")
        # reshape to [B*L_out, kernel_size], because each patch is processed independently
        B_, KL, Lout = patches.shape
        patches = patches.permute(0, 2, 1).contiguous()  # [B, L_out, kernel_size]
        patches = patches.view(-1, self.kernel_size)  # [B*L_out, kernel_size]

        # Normalize patches to [0,1] (robust amplitudes) — optional but recommended
        # You can adapt normalization to your domain (z-score, min-max per batch, etc.)
        # Here we use a simple tanh squash to keep inputs bounded without extra statistics.
        patches_norm = torch.tanh(patches)

        # Apply the quantum patch; TorchLayer supports batched leading dims
        # If your PennyLane/Torch versions have trouble with batching, you can for-loop as fallback.
        print(f"Applying quantum patch to {patches_norm.shape[0]} patches...")
        print(f"{patches_norm.shape=}")
        print(f"{x[0, 0:10]=}")
        print(f"{patches[0:3]=}")
        # manually loop over batch dimensions
        qfeat = [self.qpatch(p) for p in patches_norm]
        qfeat = torch.stack(qfeat, dim=0)
        qfeat = qfeat + self.bias  # affine tweak

        # Reshape to [B, L_out, C_out] -> permute to [B, C_out, L_out]
        qfeat = qfeat.view(B, Lout, self.out_channels).permute(0, 2, 1).contiguous()
        return qfeat


class QuanvCNN1D(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        self.seq_len = hparams["input_length"]
        self.kernel_size = hparams["kernel_size"]
        self.stride = hparams["stride"]
        self.conv1_channels = hparams.get("conv1_channels", 1)
        self.quanvolution1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.conv1_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.dropout_val = hparams["dropout"]
        sample_input = torch.randn(
            1, 1, self.seq_len
        )  # batch_size=1, channels=1, seq_len
        self.conv1_out_len = self.quanvolution1(sample_input).shape[-1]

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
        x = self.quanvolution1(x)  # shape (batch_size, conv1_channels, conv_out_len)
        # collapse channels dimension
        x = self.fc_channel_collapse(x.transpose(1, 2)).squeeze(
            2
        )  # shape (batch_size, conv_out_len)
        x = self.fc_final(x)  # shape (batch_size, output_dim)
        return x


if __name__ == "__main__":
    data_save_dir = Path("data") / "gaussian_peak"
    data_save_path_train = data_save_dir / "train.h5"
    data_save_path_test = data_save_dir / "test.h5"

    with h5.File(data_save_path_train, "r") as f:
        X_train_full = f["X"][:].squeeze(2)
        y_train_full = f["y"][:]

    with h5.File(data_save_path_test, "r") as f:
        X_test_full = f["X"][:].squeeze(2)
        y_test_full = f["y"][:]

    print(f"X_train shape: {X_train_full.shape}")
    print(f"y_train shape: {y_train_full.shape}")
    print(f"X_test shape: {X_test_full.shape}")
    print(f"y_test shape: {y_test_full.shape}")

    n_train = 150
    n_val = 250  # validation is much faster than training
    n_test = 500  # testing is much faster than training
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_val = X_train_full[n_train : n_train + n_val]
    y_val = y_train_full[n_train : n_train + n_val]
    X_test = X_test_full[:n_test]
    y_test = y_test_full[:n_test]

    seq_len = X_train.shape[1]

    # normalize data to [0, 1]
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_val = (X_val - X_val.min()) / (X_val.max() - X_val.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

    # test the quantum patch layer
    qlayer = make_quantum_patch_torchlayer(
        patch_size=4,
        out_channels=4,
        depth=2,
        dev_name="default.qubit",
        shots=None,
    )

    patch = X_train[0, :4]
    patch = torch.tensor(patch, dtype=torch.float32)
    patch_out = qlayer(patch)
    print(f"Input patch: {patch}")
    print(f"Output features: {patch_out}")

    # test the Quanv1D layer
    qlayer1d = Quanv1D(
        in_channels=1,
        out_channels=4,
        kernel_size=4,
        stride=2,
        padding=1,
        depth=2,
    )

    qlayer1d.eval()
    qinput = torch.tensor(X_train[:2].reshape(2, 1, seq_len), dtype=torch.float32)
    qoutput = qlayer1d(qinput)
    print(f"Quanv1D input shape: {qinput.shape}")
    print(f"Quanv1D output shape: {qoutput.shape}")
    print(f"Quanv1D output: {qoutput}")

    hparams = {
        "kernel_size": (kernel_size := 4),  # =1D kernel size
        "input_length": seq_len,
        "stride": 3,  # stride for the convolution
        "output_dim": 1,  # output dimension of the final FC NN
        "hidden_layers_dims": [10, 5],
        "dropout": 0.1,  # dropout bate
        "conv1_channels": 4,
    }
    QCNN_model = QuanvCNN1D(hparams)
    QCNN_model.eval()
    QCNN_input = torch.tensor(X_train[:2].reshape(2, 1, seq_len), dtype=torch.float32)
    QCNN_output = QCNN_model(QCNN_input)
    print(f"QCNN input shape: {QCNN_input.shape}")
    print(f"QCNN output shape: {QCNN_output.shape}")
