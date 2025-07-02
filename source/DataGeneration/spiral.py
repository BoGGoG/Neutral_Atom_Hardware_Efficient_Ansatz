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


def spiral_classification_data(
    num_points=100,
    noise_std: float = 0.05,
    revolutions: float = 1.5,
    radius_scale: float = 1.0,
):
    """
    Class 0 and Class 1 are interleaved spirals.
    Each class has `num_points` samples.
    """
    points = []
    classes = []

    total_points = num_points * 2
    n_attempts = 0
    max_attempts = 10 * total_points  # safety cap

    while len(points) < total_points and n_attempts < max_attempts:
        class_label = len(points) % 2  # alternate between class 0 and 1
        idx_in_class = len(points) // 2

        # Normalize t between 0 and 1
        t = idx_in_class / num_points
        angle = t * revolutions * 2 * np.pi + (np.pi if class_label == 1 else 0)
        r = t * radius_scale

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Add noise after class decision
        x += np.random.normal(0, noise_std)
        y += np.random.normal(0, noise_std)

        points.append([x, y])
        classes.append(class_label)
        n_attempts += 1

    points = np.array(points)
    classes = np.array(classes)

    # Shuffle
    idxs = np.random.permutation(len(points))
    points = points[idxs]
    classes = classes[idxs]

    print(f"Classes distribution: {Counter(classes)}")
    return points, classes


if __name__ == "__main__":
    data_save_dir = Path("data") / "spiral_2"
    data_save_path_train = data_save_dir / "train.h5"
    data_save_path_test = data_save_dir / "test.h5"
    os.makedirs(data_save_dir, exist_ok=True)

    points, classes = spiral_classification_data(
        revolutions=3, radius_scale=3.0, num_points=2_000, noise_std=0.15
    )
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
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("spiral Classification Data")
    plt.legend()
    plt.savefig(fig_path := data_save_dir / "spiral_classification_data.png")
    print(f"Figure saved to {fig_path}")
