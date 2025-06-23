import os
from typing import Callable, Optional
from pathlib import Path
import shutil
import tqdm

import numpy as np
import torch


def train_loop(
    dl_train,
    dl_val,
    model_run_fun: Callable,
    params: dict,
    data_save_file: Optional[Path] = None,
    epochs: int = 2,
    lr: float = 0.01,
):
    """
    Parameters
    - dl_train: DataLoader for training data
    - dl_val: DataLoader for validation data
    - model_run_fun: Function to run the model
    - params: Dictionary containing model parameters
    - data_save_file: Optional file path to save training data. Each element contains a rather lengthy simulation and I want to save the data and later train a neural net on it.
    """
    print("Starting training loop...")
    if data_save_file is None:
        print("No data_save_file provided, generated data will not be saved.")
    else:
        os.makedirs(data_save_file.parent, exist_ok=True)
        shutil.copy(Path("source/model.py"), data_save_file.parent)
        shutil.copy(Path("source/train_loop.py"), data_save_file.parent)
        shutil.copy(Path("scripts/model1_hpopt.py"), data_save_file.parent)
        print(
            f"Copied source files to data_save_file directory {data_save_file.parent}"
        )

    positions = params["positions"]
    local_pulses_omega = params["local_pulses_omega"]
    local_pulses_delta = params["local_pulses_delta"]
    global_pulse_omega = params["global_pulse_omega"]
    global_pulse_delta = params["global_pulse_delta"]
    local_pulse_duration = params["local_pulse_duration"]
    global_pulse_duration = params["global_pulse_duration"]
    embed_pulse_duration = params["embed_pulse_duration"]
    sampling_rate = params["sampling_rate"]
    protocol = params.get("protocol", None)
    if protocol is None:
        print("Warning: No protocol for pulses provided, using default 'min-delay'.")
        protocol = "min-delay"

    parameters = [
        positions,
        local_pulses_omega,
        local_pulses_delta,
        global_pulse_omega,
        global_pulse_delta,
    ]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    loss_fn = torch.nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification

    losses_hist = []  # list to store losses for each batch
    accuracies_hist = []
    positions_hist = [positions.clone().detach().numpy()]
    local_pulses_omega_hist = [local_pulses_omega.clone().detach().numpy()]
    local_pulses_delta_hist = [local_pulses_delta.clone().detach().numpy()]
    global_pulse_omega_hist = [global_pulse_omega.clone().detach().numpy()]
    global_pulse_delta_hist = [global_pulse_delta.clone().detach().numpy()]
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        for i_batch, batch in enumerate(dl_train):
            x_batch, y_batch = batch
            optimizer.zero_grad()
            l = torch.tensor(0.0, requires_grad=False)
            y_pred_batch = []
            for i in range(len(x_batch)):
                print(f"\tProcessing sample {i+1}/{len(x_batch)}", end="\r")
                out, states = model_run_fun(
                    x_batch[i],
                    positions,
                    local_pulses_omega,
                    local_pulses_delta,
                    global_pulse_omega,
                    global_pulse_delta,
                    local_pulse_duration=local_pulse_duration,
                    global_pulse_duration=global_pulse_duration,
                    embed_pulse_duration=embed_pulse_duration,
                    draw_reg_seq=False,
                    sampling_rate=sampling_rate,
                )
                loss = loss_fn(
                    out, y_batch[i : i + 1].to(torch.float64)
                )  # needs float64 here
                y_pred_batch.append(out.item())
                l += loss
                # save parameters and output to file
                if data_save_file is not None:
                    with open(data_save_file, "a") as f:
                        # f.write(f"{positions.tolist()};{local_pulses_omega.tolist()};{local_pulses_delta.tolist()}; {global_pulse_omega.item()};{global_pulse_delta.item()};{states[-1].tolist()};{out.item()};{x_batch[i].tolist()};{y_batch[i].item()}\n")
                        f.write(
                            f"{sampling_rate};{local_pulse_duration};{global_pulse_duration};{embed_pulse_duration};"
                            f"{positions.tolist()};{local_pulses_omega.tolist()};{local_pulses_delta.tolist()};"
                            f"{global_pulse_omega.item()};{global_pulse_delta.item()};{states[-1].tolist()};"
                            f"{out.item()};{x_batch[i].tolist()};{y_batch[i].item()}\n"
                        )

            l /= len(x_batch)
            l.backward()
            optimizer.step()
            losses_hist.append(l.item())
            positions_hist.append(positions.clone().detach().numpy())
            # calculate accuracy for the batch
            targets = y_batch.numpy()
            predictions = np.array(y_pred_batch)
            predictions = np.where(
                predictions > 0.5, 1, 0
            )  # Convert to binary predictions
            correct_predictions = np.sum(predictions == targets)
            accuracy = correct_predictions / len(targets)
            accuracies_hist.append(accuracy)
            print(
                f"\tBatch {i_batch+1}/{len(dl_train)}, Loss: {l.item():.4f}, Accuracy: {accuracy:.4f}"
            )
        local_pulses_omega_hist.append(local_pulses_omega.clone().detach().numpy())
        local_pulses_delta_hist.append(local_pulses_delta.clone().detach().numpy())
        global_pulse_omega_hist.append(global_pulse_omega.clone().detach().numpy())
        global_pulse_delta_hist.append(global_pulse_delta.clone().detach().numpy())

    ### validation at the end of the training loop
    val_losses = []
    val_correct_predictions = []
    print(
        f"Starting validation: {len(dl_val)} batches a {dl_val.batch_size} samples each (total len: {len(dl_val.dataset)})"
    )
    with torch.no_grad():
        for batch in dl_val:
            x_batch, y_batch = batch
            l = torch.tensor(0.0, requires_grad=False)
            y_pred_batch = []
            for i in tqdm(range(len(x_batch))):
                out, states = model_run_fun(
                    x_batch[i],
                    positions,
                    local_pulses_omega,
                    local_pulses_delta,
                    global_pulse_omega,
                    global_pulse_delta,
                    local_pulse_duration=local_pulse_duration,
                    global_pulse_duration=global_pulse_duration,
                    embed_pulse_duration=embed_pulse_duration,
                    draw_reg_seq=False,
                    sampling_rate=sampling_rate,
                    protocol=protocol,
                )
                loss = loss_fn(
                    out, y_batch[i : i + 1].to(torch.float64)
                )  # needs float64 here
                y_pred_batch.append(out.item())
                l += loss
            l /= len(x_batch)
            predictions = np.array(y_pred_batch)
            predictions = np.where(predictions > 0.5, 1, 0)
            correct_predictions = predictions == y_batch.numpy()
            val_losses.append(l.item())
            val_correct_predictions.append(correct_predictions)
    val_accuracy = np.mean(val_correct_predictions)
    val_loss = np.mean(val_losses)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    out_dic = {
        "train_loss_hist": losses_hist,
        "train_accuracy_hist": accuracies_hist,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
    }
    out_params = {
        "local_pulses_omega": local_pulses_omega_hist,
        "local_pulses_delta": local_pulses_delta_hist,
        "global_pulse_omega": global_pulse_omega_hist,
        "global_pulse_delta": global_pulse_delta_hist,
        "positions": positions_hist,
    }

    return out_dic, out_params
