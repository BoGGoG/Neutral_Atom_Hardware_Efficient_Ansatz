import numpy as np
from sklearn.model_selection import train_test_split
import h5py as h5
from pathlib import Path
import matplotlib.pyplot as plt
import os


def generate_sin_trend_data(
    n_samples, seq_len, noise=0.1, seed=42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sinusoidal data + linear trend + phase
    """
    np.random.seed(seed)
    X = []
    y = []
    for _ in range(n_samples):
        phase = np.random.uniform(0, 2 * np.pi)  # random phase shift
        freq = np.random.uniform(0.1, 3.0)  # frequency between 0.1 and 1.0
        slope = np.random.uniform(-0.5, 0.5)  # random slope for linear trend
        intercept = np.random.uniform(-1.0, 1.0)  # random intercept for linear trend
        x = np.linspace(0, 6 * np.pi, seq_len)
        seq = np.sin(freq * x + phase) + slope * x + intercept
        seq += np.random.normal(0, noise, seq_len)  # add some noise
        X.append(seq)
        y.append([freq, slope, intercept, phase])
    X = np.array(X).reshape(n_samples, seq_len, 1)  # shape (n_samples, seq_len, 1)
    y = np.array(y).reshape(n_samples, 4)  # shape (n_samples, 4)
    return X, y


if __name__ == "__main__":
    data_save_dir = Path("data") / "sin_trend"
    os.makedirs(data_save_dir, exist_ok=True)
    data_save_path_train = data_save_dir / "train.h5"
    data_save_path_test = data_save_dir / "test.h5"
    n_samples = 5_000
    seq_len = 21
    noise = 0.10

    X, y_true = generate_sin_trend_data(n_samples, seq_len, noise=noise, seed=42)

    # train test split
    percentage = 0.8
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=1 - percentage, random_state=42
    )

    # # normalize X to [0, 1]
    # X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    # X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
    print(f"Generated data shapes: {X_train.shape=}, {y_train.shape=}")
    print(f"Generated data shapes: {X_test.shape=}, {y_test.shape=}")

    # export the data to a .h5 file
    with h5.File(data_save_path_train, "w") as f:
        assert len(X_train) == len(y_train), "X and y must have the same length"
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)
        # add description attributes for y
        f["y"].attrs["description"] = "[frequency, slope, intercept, phase]"
        print(f"Train data saved to {data_save_path_train}")
    with h5.File(data_save_path_test, "w") as f:
        assert len(X_test) == len(y_test), "X and y must have the same length"
        f.create_dataset("X", data=X_test)
        f.create_dataset("y", data=y_test)
        f["y"].attrs["description"] = "[frequency, slope, intercept, phase]"
        print(f"Test data saved to {data_save_path_test}")

    # plot some samples
    plot_path = data_save_dir / "samples.png"
    plt.figure(figsize=(10, 5))
    for i in range(7):
        plt.plot(
            X_train[i].flatten(),
            label=f"Sample {i+1}, pos = {y_train[i][0]:.2f}, height = {y_train[i][1]:.2f}, width = {y_train[i][2]:.2f}",
        )
    plt.title("Generated Gaussian Peak Data Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(plot_path)
