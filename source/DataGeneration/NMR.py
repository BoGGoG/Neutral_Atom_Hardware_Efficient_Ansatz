import numpy as np
from sklearn.model_selection import train_test_split
import h5py as h5
from pathlib import Path
import matplotlib.pyplot as plt
import os
import requests


# def generate_gaussian_peak_data(
#     n_samples, seq_len, noise=0.1, seed=42
# ) -> tuple[np.ndarray, np.ndarray]:
#     np.random.seed(seed)
#     X = []
#     y = []
#     for _ in range(n_samples):
#         peak_pos = np.random.uniform(0, 1)  # random peak position in [0, 1]
#         peak_height = np.random.uniform(0.5, 1.5)  # random peak height
#         sigma = np.random.uniform(0.01, 0.1)  # random peak width
#         x_range = np.linspace(0, 1, seq_len)
#         yy = (
#             peak_height
#             * np.exp(-((x_range - peak_pos) ** 2) / (2 * sigma**2))
#             / (sigma * np.sqrt(2 * np.pi))
#         )
#         X.append(yy + np.random.normal(0, noise, seq_len))  # add some noise
#         y.append([peak_pos, peak_height, sigma])  # target is the peak height
#     X = np.array(X).reshape(-1, seq_len, 1)  # shape (n_samples, seq_len, 1)
#     y = np.array(y)
#     return X, y


def download_NMR_data(data_save_dir: Path):
    print("Downloading files off google drive...")

    f_prefix = data_save_dir / "tmp" / "gauss"
    os.makedirs(f_prefix, exist_ok=True)

    # data for model creation
    gauss_train_y = f_prefix / "_mat_info_model.txt"
    gauss_train_X_real = f_prefix / "_echos_model_r.txt"  # real part of echos
    gauss_train_X_imaginary = f_prefix / "_echos_model_i.txt"  # imaginary part of echos

    asdf = "https://drive.google.com/uc?export=download&id="

    r = requests.get(asdf + "1J8CcJVQRpzSwue1vuHV9uB0bngdDrKCY", allow_redirects=True)
    open(gauss_train_y, "wb").write(r.content)
    r = requests.get(asdf + "1lBWcwF--1rrB8KCyCd0-5ZnPIjRrWkHg", allow_redirects=True)
    open(gauss_train_X_real, "wb").write(r.content)
    r = requests.get(asdf + "1O7KKL-SW3vHePoRNk8YfLzX82wf2Z5ul", allow_redirects=True)
    open(gauss_train_X_imaginary, "wb").write(r.content)

    # data for submission of final model
    gauss_test_X_real = f_prefix / "_echos_eval_r.txt"  # real part of echos
    gauss_test_X_imaginary = f_prefix / "_echos_eval_i.txt"  # imaginary part of echos

    r = requests.get(asdf + "1prIrtO7XJs3PBe1MZiWUxK3VUkrChVvz", allow_redirects=True)
    open(gauss_test_X_real, "wb").write(r.content)
    r = requests.get(asdf + "1vbKcuxe6z8cRGQdTqj_Q2u5Oow0D9hbU", allow_redirects=True)
    open(gauss_test_X_imaginary, "wb").write(r.content)

    # now repeat, but for rrky type function

    f_prefix = data_save_dir / "tmp" / "rrky"
    os.makedirs(f_prefix, exist_ok=True)

    # data for model training
    rrky_train_y = f_prefix / "_mat_info_model.txt"
    rrky_train_X_real = f_prefix / "_echos_model_r.txt"  # real part of echos
    rrky_train_X_imaginary = f_prefix / "_echos_model_i.txt"  # imaginary part of echos
    r = requests.get(asdf + "1lS9AJ3sUFI4cfM5jQj618x4shoaJMXVo", allow_redirects=True)
    open(rrky_train_y, "wb").write(r.content)
    r = requests.get(asdf + "1J21bKy8FTjoaGzHVdLXlWAao2UiWO7ml", allow_redirects=True)
    open(rrky_train_X_real, "wb").write(r.content)
    r = requests.get(asdf + "1nf3Y_FcJJEWXJbjwREAkgcnVz2tDA__I", allow_redirects=True)
    open(rrky_train_X_imaginary, "wb").write(r.content)

    # data for submission of final model
    rrky_test_X_real = f_prefix / "_echos_eval_r.txt"  # real part of echos
    rrky_test_X_imaginary = f_prefix / "_echos_eval_i.txt"  # imaginary part of echos

    r = requests.get(asdf + "1Q46o_RnYZFWEjMVVF5m1VBI9HCltspyY", allow_redirects=True)
    open(rrky_test_X_real, "wb").write(r.content)
    r = requests.get(asdf + "1-z2ADFrBlEhXN5Z_LHiRLA4Nds_9uvQq", allow_redirects=True)
    open(rrky_test_X_imaginary, "wb").write(r.content)
    print("Done with file downloads")


def get_data(data_save_dir: Path = Path("data") / "NMR_data") -> dict:
    f_prefix_gauss = data_save_dir / "tmp" / "gauss"
    gauss_train_y = f_prefix_gauss / "_mat_info_model.txt"
    gauss_train_X_real = f_prefix_gauss / "_echos_model_r.txt"  # real part of echos
    gauss_train_X_imaginary = (
        f_prefix_gauss / "_echos_model_i.txt"
    )  # imaginary part of echos
    gauss_test_X_real = f_prefix_gauss / "_echos_eval_r.txt"  # real part of echos
    gauss_test_X_imaginary = (
        f_prefix_gauss / "_echos_eval_i.txt"
    )  # imaginary part of echos

    f_prefix_rrky = data_save_dir / "tmp" / "RRKY"
    rrky_train_y = f_prefix_rrky / "_mat_info_model.txt"
    rrky_train_X_real = f_prefix_rrky / "_echos_model_r.txt"  # real part of echos
    rrky_train_X_imaginary = (
        f_prefix_rrky / "_echos_model_i.txt"
    )  # imaginary part of echos
    rrky_test_X_real = f_prefix_rrky / "_echos_eval_r.txt"  # real part of echos
    rrky_test_X_imaginary = (
        f_prefix_rrky / "_echos_eval_i.txt"
    )  # imaginary part of echos

    gauss_train_X_real = np.loadtxt(gauss_train_X_real, comments="#", delimiter=None, unpack=False)
    gauss_train_X_imaginary = np.loadtxt(gauss_train_X_imaginary, comments="#", delimiter=None, unpack=False)
    # combine X_real and X_imaginary into a single array of shape (n_samples, seq_len, 2)
    gauss_train_X = np.dstack((gauss_train_X_real, gauss_train_X_imaginary))
    gauss_train_y = np.loadtxt(gauss_train_y, comments="#", delimiter=None, unpack=False)

    # train/val split
    gauss_train_X, gauss_val_X, gauss_train_y, gauss_val_y = train_test_split(gauss_train_X, gauss_train_y, test_size=0.2, random_state=42)
    

    gauss_test_X_real = np.loadtxt(gauss_test_X_real, comments="#", delimiter=None, unpack=False)
    gauss_test_X_imaginary = np.loadtxt(gauss_test_X_imaginary, comments="#", delimiter=None, unpack=False)
    gauss_test_X = np.dstack((gauss_test_X_real, gauss_test_X_imaginary))

    # rrky
    rrky_train_X_real = np.loadtxt(rrky_train_X_real, comments="#", delimiter=None, unpack=False)
    rrky_train_X_imaginary = np.loadtxt(rrky_train_X_imaginary, comments="#", delimiter=None, unpack=False)
    # combine X_real and X_imaginary into a single array of shape (n_samples, seq_len, 2)
    rrky_train_X = np.dstack((rrky_train_X_real, rrky_train_X_imaginary))
    rrky_train_y = np.loadtxt(rrky_train_y, comments="#", delimiter=None, unpack=False)

    rrky_train_X, rrky_val_X, rrky_train_y, rrky_val_y = train_test_split(rrky_train_X, rrky_train_y, test_size=0.2, random_state=42)

    rrky_test_X_real = np.loadtxt(rrky_test_X_real, comments="#", delimiter=None, unpack=False)
    rrky_test_X_imaginary = np.loadtxt(rrky_test_X_imaginary, comments="#", delimiter=None, unpack=False)
    rrky_test_X = np.dstack((rrky_test_X_real, rrky_test_X_imaginary))

    out = {
        "gauss":
        {
            "train": [gauss_train_X, gauss_train_y],
            "val": [gauss_val_X, gauss_val_y],
            "test": gauss_test_X,
        },
        "rrky": {
            "train": [rrky_train_X, rrky_train_y],
            "val": [rrky_val_X, rrky_val_y],
            "test": rrky_test_X,
        }
    }

    return out


if __name__ == "__main__":
    data_save_dir = Path("data") / "NMR_data"
    os.makedirs(data_save_dir, exist_ok=True)
    # download_NMR_data(data_save_dir)

    data_save_path_gauss_train = data_save_dir / "gauss_train.h5"
    data_save_path_gauss_val = data_save_dir / "gauss_val.h5"
    data_save_path_gauss_test = data_save_dir / "gauss_test.h5"
    data_save_path_rrky_train = data_save_dir / "rrky_train.h5"
    data_save_path_rrky_val = data_save_dir / "rrky_val.h5"
    data_save_path_rrky_test = data_save_dir / "rrky_test.h5"

    data = get_data()
    print("Data shapes:")
    for key, value in data.items():
        print(f"{key} train: {value['train'][0].shape}, {value['train'][1].shape}")

    # save the data to .h5 files

    with h5.File(data_save_path_gauss_train, "w") as f:
        assert len(data["gauss"]["train"][0]) == len(data["gauss"]["train"][1]), "X and y must have the same length"
        f.create_dataset("X", data=data["gauss"]["train"][0])
        f.create_dataset("y", data=data["gauss"]["train"][1])
        print(f"Train data saved to {data_save_path_gauss_train}")
        
    with h5.File(data_save_path_gauss_val, "w") as f:
        assert len(data["gauss"]["val"][0]) == len(data["gauss"]["val"][1]), "X and y must have the same length"
        f.create_dataset("X", data=data["gauss"]["val"][0])
        f.create_dataset("y", data=data["gauss"]["val"][1])
        print(f"Val data saved to {data_save_path_gauss_val}")

    with h5.File(data_save_path_rrky_train, "w") as f:
        assert len(data["rrky"]["train"][0]) == len(data["gauss"]["train"][1]), "X and y must have the same length"
        f.create_dataset("X", data=data["rrky"]["train"][0])
        f.create_dataset("y", data=data["rrky"]["train"][1])
        print(f"Train data saved to {data_save_path_rrky_train}")
        
    with h5.File(data_save_path_rrky_val, "w") as f:
        assert len(data["rrky"]["val"][0]) == len(data["gauss"]["val"][1]), "X and y must have the same length"
        f.create_dataset("X", data=data["rrky"]["val"][0])
        f.create_dataset("y", data=data["rrky"]["val"][1])
        print(f"Val data saved to {data_save_path_rrky_val}")

    # plot some samples
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    plot_path = data_save_dir / "samples_gauss.png"
    plt.figure(figsize=(10, 5))
    for i in range(7):
        y_str = [f"{v:.2f}" for v in data["gauss"]["train"][1][i]]
        plt.plot(data["gauss"]["train"][0][i][:, 0],
                 label=f"Real of sample {i + 1}, y = {y_str}", color=colors[i % len(colors)])
        plt.plot(data["gauss"]["train"][0][i][:, 1], linestyle='--', color=colors[i % len(colors)])
    plt.title("Generated Gaussian Peak Data Samples (dashed=imag. part)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    plot_path = data_save_dir / "samples_rrky.png"
    plt.figure(figsize=(10, 5))
    for i in range(7):
        y_str = [f"{v:.2f}" for v in data["rrky"]["train"][1][i]]
        plt.plot(data["rrky"]["train"][0][i][:, 0],
                 label=f"Real of sample {i + 1}, y = {y_str}", color=colors[i % len(colors)])
        plt.plot(data["rrky"]["train"][0][i][:, 1], linestyle='--', color=colors[i % len(colors)])
    plt.title("Generated Rrky Data Samples (dashed=imag. part)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


    #
    # X, y_true = generate_gaussian_peak_data(n_samples, seq_len, noise=noise, seed=42)
    #
    # # train test split
    # percentage = 0.8
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y_true, test_size=1 - percentage, random_state=42
    # )
    #
    # # # normalize X to [0, 1]
    # # X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    # # X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())
    # print(f"Generated data shapes: {X_train.shape=}, {y_train.shape=}")
    # print(f"Generated data shapes: {X_test.shape=}, {y_test.shape=}")
    #
    # # export the data to a .h5 file
    # with h5.File(data_save_path_train, "w") as f:
    #     assert len(X_train) == len(y_train), "X and y must have the same length"
    #     f.create_dataset("X", data=X_train)
    #     f.create_dataset("y", data=y_train)
    #     print(f"Train data saved to {data_save_path_train}")
    # with h5.File(data_save_path_test, "w") as f:
    #     assert len(X_test) == len(y_test), "X and y must have the same length"
    #     f.create_dataset("X", data=X_test)
    #     f.create_dataset("y", data=y_test)
    #     print(f"Test data saved to {data_save_path_test}")
    #
    # # plot some samples
    # plot_path = data_save_dir / "samples.png"
    # plt.figure(figsize=(10, 5))
    # for i in range(7):
    #     plt.plot(
    #         X_train[i].flatten(),
    #         label=f"Sample {i + 1}, pos = {y_train[i][0]:.2f}, height = {y_train[i][1]:.2f}, width = {y_train[i][2]:.2f}",
    #     )
    # plt.title("Generated Gaussian Peak Data Samples")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.savefig(plot_path)
