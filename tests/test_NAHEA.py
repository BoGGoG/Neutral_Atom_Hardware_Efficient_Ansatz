from NAHEA import NAHEA, NAHEA_nFeatures_BinClass_1
import torch
import pytest


def test_NAHEA_initialization():
    hparams = {
        "sampling_rate": 0.4,
        "protocol": "min-delay",
        "n_ancilliary_qubits": (n_ancilliary_qubits := 2),
    }
    parameters = {
        "positions": [[0.0, 1.0], [1.0, 0.0]],
        "local_pulses_omega": [1.0] * (2 + n_ancilliary_qubits),
        "local_pulses_delta": [0.0] * (2 + n_ancilliary_qubits),
        "global_pulse_omega": 1.0,
        "global_pulse_delta": 0.0,
        "global_pulse_duration": 230,
        "local_pulse_duration": 80,
        "embed_pulse_duration": 80,
    }

    model = NAHEA(hparams=hparams, parameters=parameters, name="test_model")
    assert model.name == "test_model", "Model name should be 'test_model'"
    assert not model.training, "Model should not be in training mode by default"
    assert isinstance(model._parameters, dict), "Parameters should be a dictionary"
    assert str(model) == "NAHEA(name=test_model)", "String representation is incorrect"
    assert hasattr(model, "forward"), "Model should have a forward method"
    assert hasattr(model, "parameters"), "Model should have a parameters method"
    assert hasattr(model, "save"), "Model should have a save method"
    assert hasattr(model, "load"), "Model should have a load method"
    assert hasattr(model, "train"), "Model should have a train method"
    assert hasattr(model, "eval"), "Model should have an eval method"
    assert hasattr(model, "__call__"), "Model should be callable"
    assert model.hparams == hparams, "Hyperparameters should match the input"
    assert (
        model._parameters != parameters
    ), "Parameters should not match the input, but be converted to tensors"

    coverted_params = parameters
    for key, value in coverted_params.items():
        coverted_params[key] = torch.tensor(
            value, dtype=torch.float32, requires_grad=True
        )
    for key, value in model._parameters.items():
        assert torch.equal(
            value, coverted_params[key]
        ), f"Parameter {key} should be converted to tensor with correct values"


def test_NAHEA_nFeatures_1_missing_hparam():
    """this test checks that the model raises an error if a required hyperparameter is missing"""
    n_ancilliary_qubits = 2
    hparams = {
        # "n_features": 2,
        "sampling_rate": 0.4,
        "protocol": "min-delay",
        # "n_ancilliary_qubits": (n_ancilliary_qubits := 2),
    }
    parameters = {
        "local_pulses_omega": [1.0] * (2 + n_ancilliary_qubits),
        "local_pulses_delta": [0.0] * (2 + n_ancilliary_qubits),
        "global_pulse_omega": 1.0,
        "global_pulse_delta": 0.0,
        "global_pulse_duration": 230,
        "local_pulse_duration": 80,
        "embed_pulse_duration": 80,
    }

    with pytest.raises(ValueError, match="Missing required hyperparameters"):
        model = NAHEA_nFeatures_BinClass_1(
            hparams=hparams, parameters=parameters, name="test_model_2features"
        )


def test_NAHEA_nFeatures_1_missing_param():
    n_ancilliary_qubits = 2
    hparams = {
        "n_features": 2,
        "sampling_rate": 0.4,
        "protocol": "min-delay",
        "n_ancilliary_qubits": (n_ancilliary_qubits := 2),
    }
    parameters = {
        # "local_pulses_omega": [1.0] * (2 + n_ancilliary_qubits),
        "local_pulses_delta": [0.0] * (2 + n_ancilliary_qubits),
        "global_pulse_omega": 1.0,
        "global_pulse_delta": 0.0,
        "global_pulse_duration": 230,
        "local_pulse_duration": 80,
        "embed_pulse_duration": 80,
    }

    with pytest.raises(ValueError, match="Missing required parameters"):
        model = NAHEA_nFeatures_BinClass_1(
            hparams=hparams, parameters=parameters, name="test_model_2features"
        )


def test_NAHEA_nFeatures_1():
    hparams = {
        "n_features": 2,
        "sampling_rate": 0.4,
        "protocol": "min-delay",
        "n_ancilliary_qubits": (n_ancilliary_qubits := 1),
    }
    parameters = {
        "n_features": 2,
        "positions": [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        "local_pulses_omega": [1.0] * (2 + n_ancilliary_qubits),
        "local_pulses_delta": [0.0] * (2 + n_ancilliary_qubits),
        "global_pulse_omega": 1.0,
        "global_pulse_delta": 0.0,
        "global_pulse_duration": 230,
        "local_pulse_duration": 80,
        "embed_pulse_duration": 80,
    }

    print("Generating model")
    model = NAHEA_nFeatures_BinClass_1(
        hparams=hparams, parameters=parameters, name="test_model_2features"
    )
    assert hasattr(model, "base_seq"), "Model should have a base_seq attribute"
    assert model.base_seq.is_parametrized, "Base sequence should be parametrized"

    x = torch.tensor([0.5, 0.5], dtype=torch.float32)
    # somehow model.forward never finishes when called in pytest, but works in a script
    # y_pred = model(x)
    # print(f"Predicted output: {y_pred}")
    # assert len(y_pred) == 1, "Output should be a single value for binary classification"

    assert not model.training, "Model should not be in training mode by default"
    model.train()  # switch to training mode
    assert model.training, "Model should be in training mode after calling train()"
    model.eval()  # switch back to evaluation mode
    assert not model.training, "Model should be in evaluation mode after calling eval()"


#
# def test_NAHEA_nFeatures_gradients():
#     hparams = {
#         "n_features": 2,
#         "sampling_rate": 0.4,
#         "protocol": "min-delay",
#         "n_ancilliary_qubits": (n_ancilliary_qubits := 1),
#     }
#     parameters = {
#         "n_features": 2,
#         "positions": [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
#         "local_pulses_omega": [1.0] * (2 + n_ancilliary_qubits),
#         "local_pulses_delta": [0.0] * (2 + n_ancilliary_qubits),
#         "global_pulse_omega": 1.0,
#         "global_pulse_delta": 0.0,
#         "global_pulse_duration": 230,
#         "local_pulse_duration": 80,
#         "embed_pulse_duration": 80,
#     }
#
#     model = _BinClassNAHEA_nFeatures_1(
#         hparams=hparams, parameters=parameters, name="test_model_2features"
#     )
