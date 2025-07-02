from NAHEA import NAHEA, NAHEA_nFeatures_BinClass_1
import torch
import pytest
from trainer import Trainer, setup_data_loaders
from pathlib import Path
import os
import copy


def setup_model():
    sampling_rate = 0.05
    hparams = {
        "n_features": 2,
        "sampling_rate": sampling_rate,
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
        "embed_pulse_duration": 50,
    }
    folder = Path("data") / "MNIST_PCA4"
    filename_train = folder / "mnist_pca4_train.h5"
    epochs = 2
    n_load = 32 * 30
    # small_size = 16 * 16
    small_size = 3
    batch_size = 2
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
        "data_save_file": None,
        "n_load": n_load,
        "small_size": small_size,
        "pca_components": pca_components,
        "train_kwargs": train_kwargs,
        "test_kwargs": test_kwargs,
        "epochs": epochs,
        "sampling_rate": sampling_rate,
    }

    model = NAHEA_nFeatures_BinClass_1(
        hparams=hparams, parameters=parameters, name="test_model_2features"
    )
    return model, train_config


def test_Trainer():
    model, train_config = setup_model()
    dls = setup_data_loaders(train_config)
    train_loader = dls["train_loader"]
    val_loader = dls["val_loader"]

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

    assert isinstance(trainer, Trainer), "Trainer instance not created successfully"
    assert hasattr(trainer, "train"), "Trainer does not have a train method"
    assert hasattr(trainer, "train_epoch"), "Trainer does not have a train_epoch method"

    model_params_pre_training = copy.deepcopy(model._parameters)
    print(f"Model parameters before training: {model_params_pre_training.values()}")
    trainer.train(epochs=train_config["epochs"])
    model_params_post_training = model._parameters
    # make sure at least one parameter has changed
    changed = False
    for v_pre, v_post in zip(
        model_params_pre_training.values(), model_params_post_training.values()
    ):
        if not torch.equal(v_pre, v_post):
            changed = True
    assert changed, "Model parameters did not change during training"
