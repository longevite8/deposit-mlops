"""ClearML Serving preprocess adapter for deposit cashflow forecasts."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


def _truthy(value: Any) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "y"}


def _extract_archive(local_file_name: str) -> str:
    if not local_file_name.endswith(".zip") or not os.path.isfile(local_file_name):
        return local_file_name

    extract_dir = os.path.splitext(local_file_name)[0]
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(local_file_name, "r") as archive:
        archive.extractall(extract_dir)
    return extract_dir


def _load_neuralforecast(model_path: str):
    import torch
    from neuralforecast import NeuralForecast

    original_torch_load = torch.load

    def project_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = project_torch_load
    try:
        return NeuralForecast.load(path=model_path)
    finally:
        torch.load = original_torch_load


def _normalize_history_frame(body: dict[str, Any]) -> pd.DataFrame:
    rows = body.get("history") or body.get("rows") or body.get("data") or []
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Request body must include non-empty 'history', 'rows', or 'data'.")

    date_column = os.getenv("SERVING_DATE_COLUMN", "date")
    target_column = os.getenv("SERVING_TARGET_COLUMN", "cashflow")
    unique_id = body.get("unique_id") or os.getenv("FORECAST_UNIQUE_ID", "Deposit_Portfolio")

    if "ds" not in df.columns:
        if date_column not in df.columns:
            raise ValueError(f"Request rows must include 'ds' or '{date_column}'.")
        df = df.rename(columns={date_column: "ds"})
    if "y" not in df.columns:
        if target_column not in df.columns:
            raise ValueError(f"Request rows must include 'y' or '{target_column}'.")
        df = df.rename(columns={target_column: "y"})
    if "unique_id" not in df.columns:
        df["unique_id"] = unique_id

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.normalize()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["unique_id", "ds", "y"])
    if df.empty:
        raise ValueError("No valid history rows remained after parsing dates and target values.")

    df = (
        df.groupby(["unique_id", "ds"], as_index=False)["y"]
        .sum()
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    df["month"] = df["ds"].dt.month
    df["weekday"] = df["ds"].dt.weekday
    return df


class Preprocess:
    """ClearML Serving adapter loaded by the serving runtime."""

    def __init__(self):
        self._engine = None
        self._selected_model_alias = os.getenv("SERVING_MODEL_ALIAS", "")
        self._last_state: dict[str, Any] = {}

    def load(self, local_file_name: str):
        model_path = _extract_archive(local_file_name)
        self._engine = _load_neuralforecast(model_path)
        self._engine._fitted = True

        selection_path = Path(model_path) / "serving_model_selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            self._selected_model_alias = selection.get("model_alias") or self._selected_model_alias
        return self

    def preprocess(self, body: dict[str, Any], state: dict, collect_custom_statistics_fn=None):
        horizon = int(body.get("horizon") or os.getenv("FORECAST_HORIZON", "30"))
        frame = _normalize_history_frame(body)
        state["horizon"] = horizon
        state["history_rows"] = len(frame)
        self._last_state = dict(state)
        if collect_custom_statistics_fn:
            collect_custom_statistics_fn({"horizon": horizon, "history_rows": len(frame)})
        return frame

    def predict(self, data):
        return self.process(data, dict(self._last_state), None)

    def process(self, data, state: dict, collect_custom_statistics_fn=None):
        if self._engine is None:
            raise RuntimeError("Forecast model is not loaded.")

        forecasts = self._engine.predict(df=data)
        model_cols = [column for column in forecasts.columns if column not in {"unique_id", "ds"}]
        if not model_cols:
            raise ValueError("Model forecast did not return prediction columns.")

        selected_model = self._selected_model_alias or model_cols[0]
        if selected_model not in model_cols:
            if _truthy(os.getenv("SERVING_FALLBACK_TO_FIRST_MODEL", "1")):
                selected_model = model_cols[0]
            else:
                raise ValueError(
                    f"Selected model '{selected_model}' was not found in forecast columns: {model_cols}"
                )

        horizon = int(state.get("horizon", os.getenv("FORECAST_HORIZON", "30")))
        result = forecasts[["unique_id", "ds", selected_model]].head(horizon).copy()
        result = result.rename(columns={selected_model: "prediction"})
        result["ds"] = result["ds"].astype(str)
        if collect_custom_statistics_fn and not result.empty:
            collect_custom_statistics_fn(
                {
                    "prediction_mean": float(result["prediction"].mean()),
                    "prediction_min": float(result["prediction"].min()),
                    "prediction_max": float(result["prediction"].max()),
                }
            )
        return {
            "model": selected_model,
            "horizon": horizon,
            "forecast": result.to_dict(orient="records"),
        }

    def postprocess(self, data, state: dict, collect_custom_statistics_fn=None):
        return data
