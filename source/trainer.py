import json
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor, is_inference, tensor

import optuna
from source.NAHEA import NAHEA_nFeatures_BinClass_1
from tqdm import tqdm
import gc


def undersample(X, y):
    classes = Counter(y)
    n_0 = classes[0]
    n_1 = classes[1]
    idxs_0 = np.where(y == 0)[0]
    idxs_1 = np.where(y == 1)[0]

    if n_0 < n_1:
        y_0 = y[idxs_0]
        X_0 = X[idxs_0]
        y_1 = y[idxs_1[: len(idxs_0)]]
        X_1 = X[idxs_1[: len(idxs_0)]]
        y = np.concatenate([y_0, y_1])
        X = np.concatenate([X_0, X_1])
    elif n_1 < n_0:
        y_0 = y[idxs_0[: len(idxs_1)]]
        X_0 = X[idxs_0[: len(idxs_1)]]
        y_1 = y[idxs_1[idxs_0]]
        X_1 = X[idxs_1[idxs_0]]
        y = np.concatenate([y_0, y_1])
        X = np.concatenate([X_0, X_1])

    # shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return X, y


def setup_data_loaders(config):
    n_load = config["n_load"]
    with h5py.File(config["filename_train"], "r") as f:
        X_train: np.ndarray = f["X_pca"][:n_load]  # type: ignore
        y_train: np.ndarray = f["y"][:n_load]  # type: ignore
    small_size = config["small_size"]
    train_kwargs = config["train_kwargs"]
    val_kwargs = config["test_kwargs"]
    test_kwargs = val_kwargs

    # only take items where y is 1 or 5
    mask = (y_train == 1) | (y_train == 5)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # convert y_train and y_test to 0 and 1 from 1 and 5
    y_train = y_train == 1  # .long()

    #   train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    X_train, y_train = undersample(X_train, y_train)
    X_val, y_val = undersample(X_val, y_val)

    # Create TensorDataset
    X_train = torch.tensor(X_train, dtype=torch.double)[:, : config["pca_components"]]
    y_train = torch.tensor(y_train, dtype=torch.double)
    X_val = torch.tensor(X_val, dtype=torch.double)[:, : config["pca_components"]]
    y_val = torch.tensor(y_val, dtype=torch.double)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    # smaller dataset with only a few samples
    small_train_dataset = torch.utils.data.TensorDataset(
        X_train[:small_size], y_train[:small_size]
    )
    small_train_loader = torch.utils.data.DataLoader(
        small_train_dataset, **train_kwargs
    )
    small_val_dataset = torch.utils.data.TensorDataset(
        X_val[:small_size], y_val[:small_size]
    )
    small_val_loader = torch.utils.data.DataLoader(small_val_dataset, **test_kwargs)
    print("loaded data")
    print(f"y_train classes: {Counter(np.array(y_train[:small_size]))}")
    print(f"y_val classes: {Counter(np.array(y_val[:small_size]))}")

    del X_train, y_train, X_val, y_val  # free memory
    print(
        f"Train dataset: {len(train_dataset)}, validation dataset: {len(val_dataset)}"
    )

    return {
        "train_loader": small_train_loader,
        "val_loader": small_val_loader,
        "test_loader": None,
    }


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader=None,
        device="cpu",
        optimizer_kwargs=None,
    ):
        self.model = model
        self.optimizer = optimizer(
            model.parameters().values(), **(optimizer_kwargs or {})
        )
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_epoch(self):
        self.model.train()
        for batch in tqdm(self.train_loader, desc="Training epoch", leave=False):
            inputs, targets = batch  # Adapt depending on your data format
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            batch_loss = tensor(0.0, requires_grad=False).to(self.device)
            batch_correct = []
            batch_accuracy = 0.0
            self.optimizer.zero_grad()
            for x, y_true in tqdm(
                zip(inputs, targets),
                total=len(inputs),
                leave=False,
                desc="Training batch",  # type: ignore
            ):
                output = self.model(x)["output"].squeeze()
                loss = self.loss_fn(output, y_true) / len(inputs)
                batch_loss += loss
                correct = (output > 0.5).long() == y_true.long()
                batch_correct.append(correct.item())
            batch_loss /= len(inputs)
            batch_accuracy = np.mean(np.array(batch_correct, dtype=np.float32))
            batch_loss.backward()
            self.optimizer.step()

        # last batch loss and accuracy
        return {"batch_loss": batch_loss.item(), "batch_accuracy": batch_accuracy}

    def validate(self) -> dict:
        if self.val_loader is None:
            print("No validation loader provided.")
            return

        self.model.eval()
        total_loss = tensor(0.0, requires_grad=False).to(self.device)
        total_correct = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_loss = tensor(0.0, requires_grad=False).to(self.device)
                for x, y_true in zip(inputs, targets):
                    y_pred = self.model(x)["output"].squeeze()
                    loss = self.loss_fn(y_pred, y_true) / len(inputs)
                    batch_loss += loss
                    correct = (y_pred > 0.5).long() == y_true.long()
                    total_correct.append(correct.item())
            total_loss += batch_loss
        total_loss /= len(self.val_loader.dataset)
        total_loss = total_loss.to("cpu")
        accuracy = torch.mean(torch.tensor(total_correct, dtype=torch.float32))

        out = {
            "loss": total_loss.item(),  # has special importance
            "accuracy": accuracy.item(),  # any other metric
        }
        return out

    def train(self, epochs):
        for epoch in tqdm(range(1, epochs + 1), desc="Training epochs"):
            train_losses = self.train_epoch()
            train_loss = train_losses["batch_loss"]
            train_accuracy = train_losses.get("batch_accuracy", None)
            # print(
            #     f"Epoch {epoch}: Train Loss = {train_loss:.4f}"
            #     + f", acc: {train_accuracy:.4f}"
            #     if train_accuracy is not None
            #     else ""
            # )

            if self.val_loader is not None:
                val_losses = self.validate()
                # print(f"           Val Loss   = {val_losses}")


if __name__ == "__main__":
    hparams = {
        "n_features": 2,
        "sampling_rate": 0.4,
        "protocol": "min-delay",
        "n_ancilliary_qubits": (n_ancilliary_qubits := 0),
    }
    parameters = {
        "positions": [[-3.6672354, 0.0], [3.6672359, 0.0]],
        "local_pulses_omega": [1.1559689, 1.6583259],
        "local_pulses_delta": [-0.76122487, 1.5434982],
        "global_pulse_omega": -0.26719406,
        "global_pulse_delta": 1.0807998,
        "global_pulse_duration": 50,
        "local_pulse_duration": 50,
        "embed_pulse_duration": 80,
    }
    folder = Path("data") / "MNIST_PCA4"
    filename_train = folder / "mnist_pca4_train.h5"
    data_save_dir = Path("generated_data") / "dev" / "1"
    os.makedirs(data_save_dir, exist_ok=True)
    data_save_file = data_save_dir / "output.csv"
    epochs = 10
    n_load = 32 * 32 * 30
    small_size = 16 * 16
    # small_size = 10
    batch_size = 8
    pca_components = 2
    logging_dir = Path("logs") / "NAHEA" / "test_model_2features"
    optuna_log_db = logging_dir / "optuna_model2.db"
    # optuna_log_db = "sqlite:///optuna/optuna_model2.db"
    train_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
    }
    test_kwargs = {
        "batch_size": 32,
        "shuffle": False,
        "num_workers": 1,
        "pin_memory": True,
    }
    train_config = {
        "filename_train": filename_train,
        "data_save_file": data_save_file,
        "n_load": n_load,
        "small_size": small_size,
        "pca_components": pca_components,
        "train_kwargs": train_kwargs,
        "test_kwargs": test_kwargs,
        "epochs": epochs,
        "sampling_rate": 0.4,
    }

    model = NAHEA_nFeatures_BinClass_1(
        hparams=hparams, parameters=parameters, name="test_model_2features"
    )
    print(model)

    dls = setup_data_loaders(train_config)
    print(f"{train_config=}")
    train_loader = dls["train_loader"]
    val_loader = dls["val_loader"]
    print(f"{len(train_loader.dataset)=}, {len(val_loader.dataset)=}")

    lr = 0.01
    model.train()
    optimizer = torch.optim.Adam
    loss_fn = torch.nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    trainer = Trainer(
        model,
        optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_kwargs={"lr": lr},
    )

    batch = next(iter(train_loader))
    # train_loss = trainer.train_epoch()
    # print(f"Train loss: {train_loss}")
    trainer.train(epochs=train_config["epochs"])
    validation_loss = trainer.validate()
    print(f"Validation loss: {validation_loss}")
