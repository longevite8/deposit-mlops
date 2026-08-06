import pytest

from business.explainability import get_tree_explainable_model


class FakeLightGBMModel:
    def predict(self, values):
        return values


def test_get_tree_explainable_model_unwraps_lightgbm_bundle():
    model = FakeLightGBMModel()
    artifact = {
        "model_type": "lightgbm",
        "forecast_horizon": 3,
        "model": model,
    }

    assert get_tree_explainable_model(artifact, "lightgbm") is model


def test_get_tree_explainable_model_rejects_neural_model():
    artifact = {
        "model_type": "nhits",
        "neural_forecast": object(),
        "scaler_y": object(),
    }

    with pytest.raises(ValueError, match="does not support model type 'nhits'"):
        get_tree_explainable_model(artifact, "nhits")


def test_get_tree_explainable_model_rejects_missing_model_key():
    with pytest.raises(ValueError, match="does not contain the 'model' key"):
        get_tree_explainable_model({"model_type": "lightgbm"}, "lightgbm")
