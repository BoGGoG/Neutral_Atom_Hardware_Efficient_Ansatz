import numpy as np
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path
from sklearn.model_selection import train_test_split
import h5py as h5
from collections import Counter


def annulus_classification_data(
    inner_radius=1.0,
    outer_radius=2.0,
    rectangle=[3, 3],
    num_points=100,
    noise_std: float = 0.05,
):
    """
    Class 0: Points between inner_radius and outer_radius (annulus).
    Class 1: Points inside inner_radius or outside outer_radius.
    Add noise after the classification.
    """
    points = []
    classes = []
    n_points = 10 * num_points  # to ensure we have enough points in both classes
    for _ in range(n_points):
        x = np.random.uniform(-rectangle[0], rectangle[0])
        y = np.random.uniform(-rectangle[1], rectangle[1])
        r_squared = x**2 + y**2
        if inner_radius**2 <= r_squared <= outer_radius**2:
            class_label = 0
        else:
            class_label = 1
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
    data_save_dir = Path("data") / "annulus"
    data_save_path_train = data_save_dir / "train.h5"
    data_save_path_test = data_save_dir / "test.h5"
    os.makedirs(data_save_dir, exist_ok=True)
    points, classes = annulus_classification_data(
        inner_radius=1.0,
        outer_radius=2.0,
        rectangle=[3, 3],
        num_points=10_000,
        noise_std=0.2,
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
    # draw the circle
    phi = np.linspace(0, 2 * np.pi, 100)
    px = np.cos(phi)
    py = np.sin(phi)
    plt.plot(px, py, color="black", label="Circle boundary 1")
    px = 2 * np.cos(phi)
    py = 2 * np.sin(phi)
    plt.plot(px, py, color="black", label="Circle boundary 2")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Circle Classification Data")
    plt.legend()
    plt.savefig(fig_path := data_save_dir / "annulus_classification_data.png")
    print(f"Figure saved to {fig_path}")
