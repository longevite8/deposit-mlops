import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from business.inference import (
    normalize_model_artifact,
    run_champion_inference,
)


def test_normalize_legacy_raw_lightgbm_model():
    raw_model = lgb.LGBMRegressor(
        n_estimators=1,
        verbosity=-1,
    )

    artifact = normalize_model_artifact(
        artifact=raw_model,
        default_forecast_horizon=3,
    )

    assert artifact["model_type"] == "lightgbm"
    assert artifact["forecast_horizon"] == 3
    assert artifact["model"] is raw_model
    assert artifact["legacy_artifact"] is True


def test_inference_accepts_lightgbm_bundle():
    class FakeLightGBM:
        def predict(self, values):
            return np.array([10.0, 20.0])

    feature_df = pd.DataFrame(
        {
            "lag_1": [1, 2],
            "lag_7": [3, 4],
        }
    )

    artifact = {
        "model_type": "lightgbm",
        "forecast_horizon": 3,
        "model": FakeLightGBM(),
    }

    prediction_df, _, _ = run_champion_inference(
        artifact=artifact,
        feature_df=feature_df,
        feature_columns=["lag_1", "lag_7"],
    )

    assert prediction_df["prediction"].tolist() == [10.0, 20.0]
    assert prediction_df["forecast_step"].tolist() == [3, 3]


def test_inference_accepts_legacy_raw_lightgbm_model():
    raw_model = lgb.LGBMRegressor(
        n_estimators=2,
        learning_rate=0.1,
        num_leaves=4,
        verbosity=-1,
    )

    training_features = pd.DataFrame(
        {
            "lag_1": [1.0, 2.0, 3.0, 4.0],
            "lag_7": [4.0, 3.0, 2.0, 1.0],
        }
    )
    training_target = np.array([10.0, 20.0, 30.0, 40.0])

    raw_model.fit(training_features, training_target)

    feature_df = training_features.iloc[:2].copy()

    prediction_df, _, _ = run_champion_inference(
        artifact=raw_model,
        feature_df=feature_df,
        feature_columns=["lag_1", "lag_7"],
        default_forecast_horizon=3,
    )

    assert len(prediction_df) == 2
    assert prediction_df["prediction"].notna().all()
    assert prediction_df["forecast_step"].tolist() == [3, 3]


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
