import time
from typing import Any

import numpy as np
import pandas as pd


def run_champion_inference(
    artifact: dict[str, Any],
    feature_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, float, float]:
    model_type = str(artifact["model_type"]).strip().lower()
    horizon = int(artifact["forecast_horizon"])

    start_time = time.perf_counter()

    if model_type == "lightgbm":
        model = artifact["model"]
        prediction = np.asarray(model.predict(feature_df[feature_columns]))

        if prediction.ndim == 2:
            prediction = prediction[:, -1]

        prediction = prediction.reshape(-1)

        if len(prediction) != len(feature_df):
            raise ValueError("LightGBM prediction count does not match feature rows.")

        output_df = feature_df.copy()
        output_df["prediction"] = prediction
        output_df["forecast_step"] = horizon

    elif model_type in {"nhits", "nbeatsx"}:
        neural_forecast = artifact["neural_forecast"]
        forecasts = neural_forecast.predict()

        prediction_column = "NHITS" if model_type == "nhits" else "NBEATSx"

        if prediction_column not in forecasts:
            raise ValueError(f"Missing prediction column {prediction_column!r}.")

        prediction = forecasts[prediction_column].to_numpy().reshape(-1)

        if len(prediction) != horizon:
            raise ValueError(f"Expected {horizon} forecasts, got {len(prediction)}.")

        output_df = forecasts[["unique_id", "ds"]].copy()
        output_df["prediction"] = prediction
        output_df["forecast_step"] = np.arange(1, horizon + 1)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    elapsed = time.perf_counter() - start_time
    forecast_count = len(output_df)
    latency_ms = elapsed / forecast_count * 1000

    return output_df, elapsed, latency_ms


def run_model_inference(model, X):
    """
    Thực hiện dự báo và đo lường hiệu năng xử lý.
    """
    start_time = time.time()
    prediction = model.predict(X)
    inference_time = time.time() - start_time

    # Tính latency (ms per sample)
    inference_latency_ms = (inference_time / len(X)) * 1000 if len(X) > 0 else 0

    return prediction, inference_time, inference_latency_ms


def calculate_prediction_statistics(prediction):
    """
    Tính toán các thông số thống kê cơ bản của bộ kết quả dự báo.
    """
    return {
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
    }


def build_output_dataframe(original_df, prediction, column_name="prediction"):
    """
    Hợp nhất kết quả dự báo vào DataFrame gốc.
    """
    output_df = original_df.copy()
    output_df[column_name] = prediction
    return output_df
