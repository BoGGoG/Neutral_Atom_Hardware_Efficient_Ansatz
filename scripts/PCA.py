from torchvision import datasets, transforms
import torch
import numpy as np
from sklearn.decomposition import PCA
import h5py
from pathlib import Path
import os

if __name__ == "__main__":
    """
    Downloads the MNIST dataset, applies PCA to reduce the dimensionality of the images,
    saves the transformed data to HDF5 files.
    """
    pca_components = 4
    folder = Path("data") / "MNIST_PCA4"
    os.makedirs(folder, exist_ok=True)
    filename_train = folder / "mnist_pca4_train.h5"
    filename_test = folder / "mnist_pca4_test.h5"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    dataset1 = datasets.MNIST("data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("data", train=False, transform=transform)
    train_kwargs = {
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
    }
    test_kwargs = {
        "batch_size": 1024,
        "shuffle": False,
        "num_workers": 1,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Convert data loaders to numpy arrays for PCA
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Extract data from train loader
    for data, target in train_loader:
        X_train.append(data.numpy().reshape(data.shape[0], -1))
        y_train.append(target.numpy())

    # Extract data from test loader
    for data, target in test_loader:
        X_test.append(data.numpy().reshape(data.shape[0], -1))
        y_test.append(target.numpy())

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    # Apply PCA
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # normalize each feature to [0, 1]
    X_train_pca = (X_train_pca - X_train_pca.min(axis=0)) / (
        X_train_pca.max(axis=0) - X_train_pca.min(axis=0)
    )
    X_test_pca = (X_test_pca - X_test_pca.min(axis=0)) / (
        X_test_pca.max(axis=0) - X_test_pca.min(axis=0)
    )

    print(f"Saving X_train_pca shape: {X_train_pca.shape} to {filename_train}")
    # export to hdf5
    with h5py.File(filename_train, "w") as f:
        dset_X_train_pca = f.create_dataset("X_pca", data=X_train_pca)
        dset_y_train = f.create_dataset("y", data=y_train)

    print(f"Saving X_test_pca shape: {X_test_pca.shape} to {filename_test}")
    with h5py.File(filename_test, "w") as f:
        dset_X_test_pca = f.create_dataset("X_pca", data=X_test_pca)
        dset_y_test = f.create_dataset("y", data=y_test)
