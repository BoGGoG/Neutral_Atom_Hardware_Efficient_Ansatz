"""
generate XOR data
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path
from sklearn.model_selection import train_test_split
import h5py as h5
from collections import Counter


def xor_classification_data(num_points=100, rectangle=[-2, 2], noise_std: float = 0.05):
    n_points = 2 * num_points  # to ensure we have enough points in both classes
    points = []
    classes = []
    for _ in range(n_points):
        x = np.random.uniform(-rectangle[0], rectangle[0])
        y = np.random.uniform(-rectangle[1], rectangle[1])
        class_label = np.array(((x > 0) ^ (y > 0))).astype(int)  # XOR condition
        x += np.random.normal(0, noise_std)
        y += np.random.normal(0, noise_std)
        points.append([x, y])
        classes.append(class_label)
    points = np.array(points)
    classes = np.array(classes)

    # from each class, take the first num_points points
    classes_0 = np.where(classes == 0)[0][:num_points]
    classes_1 = np.where(classes == 1)[0][:num_points]
    points = np.concatenate((points[classes_0], points[classes_1]))
    classes = np.concatenate((classes[classes_0], classes[classes_1]))

    # shuffle the points and classes
    idxs = np.random.permutation(len(points))
    points = points[idxs]
    classes = classes[idxs]
    print(f"Classes distribution: {Counter(classes)}")
    return points, classes


if __name__ == "__main__":
    data_save_dir = Path("data") / "xor"
    data_save_path_train = data_save_dir / "train.h5"
    data_save_path_test = data_save_dir / "test.h5"
    os.makedirs(data_save_dir, exist_ok=True)

    points, classes = xor_classification_data(
        rectangle=[2, 2], num_points=10_000, noise_std=0.2
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
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("xor Classification Data")
    plt.legend()
    plt.savefig(fig_path := data_save_dir / "XOR_classification_data.png")
    print(f"Figure saved to {fig_path}")
