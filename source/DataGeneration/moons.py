import numpy as np
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path
from sklearn.model_selection import train_test_split
import h5py as h5
from source.utils.data_utils import undersample
from collections import Counter

import numpy as np
from collections import Counter
from sklearn import datasets


def moons_classification_data(n_samples=1000, noise=0.1, random_state=None):
    """
    Generate a two-moon dataset for binary classification.
    """
    X, y = datasets.make_moons(
        n_samples=n_samples, noise=noise, random_state=random_state
    )

    # Shuffle the data
    idxs = np.random.permutation(len(X))
    X = X[idxs]
    y = y[idxs]

    print(f"Classes distribution: {Counter(y)}")

    return X, y


if __name__ == "__main__":
    data_save_dir = Path("data") / "moons"
    data_save_path_train = data_save_dir / "train.h5"
    data_save_path_test = data_save_dir / "test.h5"
    os.makedirs(data_save_dir, exist_ok=True)

    points, classes = moons_classification_data(n_samples=10_000, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        points, classes, test_size=0.2, random_state=42
    )

    # export the data to a .h5 file
    with h5.File(data_save_path_train, "w") as f:
        assert len(X_train) == len(y_train), "X and y must have the same length"
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)
        print(f"Train data saved to {data_save_path_train}")
    with h5.File(data_save_path_test, "w") as f:
        assert len(X_test) == len(y_test), "X and y must have the same length"
        f.create_dataset("X", data=X_test)
        f.create_dataset("y", data=y_test)
        print(f"Test data saved to {data_save_path_test}")

    # plot the points
    fig, ax = plt.subplots()
    plt.scatter(
        points[classes == 0, 0],
        points[classes == 0, 1],
        color="blue",
        label="Class 0",
        alpha=0.5,
    )
    plt.scatter(
        points[classes == 1, 0],
        points[classes == 1, 1],
        color="red",
        label="Class 1",
        alpha=0.5,
    )
    # draw the moons
    plt.xlim(-2, 3)
    plt.ylim(-1, 2)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("moons Classification Data")
    plt.legend()
    plt.savefig(fig_path := data_save_dir / "moons_classification_data.png")
    plt.show()
    print(f"Figure saved to {fig_path}")
