import numpy as np
import pandas as pd
import pytest

from business.inference import run_champion_inference


class FakeLightGBM:
    def predict(self, X):
        return np.arange(len(X), dtype=float)


class FakeNeuralForecast:
    def predict(self):
        return pd.DataFrame(
            {
                "unique_id": ["target", "target", "target"],
                "ds": pd.date_range("2026-01-01", periods=3),
                "NHITS": [10.0, 11.0, 12.0],
            }
        )


def test_lightgbm_uses_nested_model_from_bundle():
    features = pd.DataFrame({"lag_1": [1, 2], "lag_7": [3, 4]})
    artifact = {
        "model_type": "lightgbm",
        "forecast_horizon": 3,
        "model": FakeLightGBM(),
    }

    result, _, _ = run_champion_inference(
        artifact,
        features,
        ["lag_1", "lag_7"],
    )

    assert result["prediction"].tolist() == [0.0, 1.0]


def test_nhits_returns_configured_forecast_horizon():
    features = pd.DataFrame({"lag_1": [1]})
    artifact = {
        "model_type": "nhits",
        "forecast_horizon": 3,
        "neural_forecast": FakeNeuralForecast(),
    }

    result, _, _ = run_champion_inference(
        artifact,
        features,
        ["lag_1"],
    )

    assert len(result) == 3
    assert result["forecast_step"].tolist() == [1, 2, 3]


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model type"):
        run_champion_inference(
            {
                "model_type": "unknown",
                "forecast_horizon": 1,
            },
            pd.DataFrame({"lag_1": [1]}),
            ["lag_1"],
        )
