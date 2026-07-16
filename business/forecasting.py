"""Forecasting helpers for deposit cashflow training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DATE_COLUMN, FORECAST_UNIQUE_ID, TARGET_COLUMN


MODEL_COLUMNS = {"unique_id", "ds", "y"}
NON_PREDICTION_COLUMNS = {"unique_id", "ds", "cutoff", "y", "step", "month", "weekday"}


@dataclass(frozen=True)
class ForecastTrainingConfig:
    """Configuration for NeuralForecast model construction and training."""

    horizon: int
    input_size: int
    learning_rate: float
    max_steps: int
    loss: str
    optimizer: str
    activation: str
    cv_windows: int
    eval_horizon: int
    seed: int
    start_padding_enabled: bool = False


def set_forecast_seed(seed: int) -> None:
    """Set deterministic seeds for NumPy, Python, and Torch when available."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    try:
        from pytorch_lightning import seed_everything

        seed_everything(seed, workers=True)
    except Exception:
        pass


def prepare_forecast_frame(
    df: pd.DataFrame,
    *,
    unique_id: str = FORECAST_UNIQUE_ID,
    date_column: str = DATE_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Convert deposit feature/raw frames to NeuralForecast schema."""

    if df.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "y", "month", "weekday"])

    missing = [column for column in (date_column, target_column) if column not in df]
    if missing:
        raise ValueError(f"Missing forecast input columns: {missing}")

    forecast_df = df[[date_column, target_column]].copy()
    forecast_df = forecast_df.rename(columns={date_column: "ds", target_column: "y"})
    forecast_df["unique_id"] = unique_id
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], errors="coerce").dt.normalize()
    forecast_df["y"] = pd.to_numeric(forecast_df["y"], errors="coerce")
    forecast_df = forecast_df.dropna(subset=["ds", "y"])
    forecast_df = (
        forecast_df.groupby(["unique_id", "ds"], as_index=False)["y"]
        .sum()
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    return add_calendar_features(forecast_df)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple calendar features expected by the forecast engine."""

    working = df.copy()
    working["ds"] = pd.to_datetime(working["ds"])
    working["month"] = working["ds"].dt.month
    working["weekday"] = working["ds"].dt.weekday
    return working


def historical_exogenous_columns(df: pd.DataFrame) -> list[str]:
    """Return model input columns beyond the core time-series schema."""

    return [column for column in df.columns if column not in MODEL_COLUMNS]


def get_loss_function(loss_name: str):
    """Return a NeuralForecast loss instance."""

    from neuralforecast.losses.pytorch import MAE, MAPE, MSE, HuberLoss

    loss_map = {"MSE": MSE(), "MAE": MAE(), "Huber": HuberLoss(), "MAPE": MAPE()}
    if loss_name not in loss_map:
        raise ValueError(f"Loss '{loss_name}' not supported.")
    return loss_map[loss_name]


def get_optimizer_class(optimizer_name: str):
    """Return a Torch optimizer class by name."""

    import torch

    optimizer_map = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop,
    }
    return optimizer_map.get(optimizer_name, torch.optim.Adam)


def build_forecast_engine(
    config: ForecastTrainingConfig,
    hist_exog: list[str] | None = None,
):
    """Build the vc-style NeuralForecast engine for deposit cashflow."""

    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx, NHITS

    loss_fn = get_loss_function(config.loss)
    optimizer_cls = get_optimizer_class(config.optimizer)
    hist_exog_list = hist_exog or None

    base_config = {
        "h": config.horizon,
        "input_size": config.input_size,
        "loss": loss_fn,
        "learning_rate": config.learning_rate,
        "optimizer": optimizer_cls,
        "max_steps": config.max_steps,
        "random_seed": config.seed,
        "start_padding_enabled": config.start_padding_enabled,
        "activation": config.activation,
        "hist_exog_list": hist_exog_list,
    }
    models = [
        NHITS(alias="NHITS", **base_config),
        NBEATSx(alias="NBEATSx", **base_config),
    ]
    return NeuralForecast(models=models, freq="D")


def calculate_forecast_metrics(
    forecast_df: pd.DataFrame,
    *,
    eval_horizon: int | None = None,
) -> dict[str, float]:
    """Calculate metrics for NeuralForecast prediction or CV output."""

    df = forecast_df.copy()
    if df.empty:
        return {}

    if eval_horizon and "cutoff" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
        df["cutoff"] = pd.to_datetime(df["cutoff"])
        df["step"] = (df["ds"] - df["cutoff"]).dt.days
        df = df[df["step"] <= eval_horizon].copy()

    model_columns = [
        column
        for column in df.columns
        if column not in NON_PREDICTION_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    metrics: dict[str, float] = {}
    if not model_columns:
        return metrics

    y_true = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    valid_true = ~np.isnan(y_true)
    y_true = y_true[valid_true]
    if y_true.size == 0:
        return metrics

    total_abs_y = np.sum(np.abs(y_true))
    non_zero_mask = y_true != 0

    for model_column in model_columns:
        y_pred_all = pd.to_numeric(df[model_column], errors="coerce").to_numpy(dtype=float)
        y_pred = y_pred_all[valid_true]
        valid_pred = ~np.isnan(y_pred)
        y_true_model = y_true[valid_pred]
        y_pred = y_pred[valid_pred]
        if y_true_model.size == 0:
            continue

        pred_non_zero_mask = y_true_model != 0
        wape = (
            0.0
            if total_abs_y == 0
            else np.sum(np.abs(y_true_model - y_pred)) / np.sum(np.abs(y_true_model))
        )
        mape = (
            0.0
            if not np.any(pred_non_zero_mask)
            else np.mean(
                np.abs(
                    (y_true_model[pred_non_zero_mask] - y_pred[pred_non_zero_mask])
                    / y_true_model[pred_non_zero_mask]
                )
            )
        )
        metrics[f"{model_column}_mape"] = float(mape)
        metrics[f"{model_column}_mae"] = float(mean_absolute_error(y_true_model, y_pred))
        metrics[f"{model_column}_rmse"] = float(
            np.sqrt(mean_squared_error(y_true_model, y_pred))
        )
        metrics[f"{model_column}_wape"] = float(wape)
        if y_true_model.size >= 2:
            metrics[f"{model_column}_r2"] = float(r2_score(y_true_model, y_pred))

    primary = select_primary_metric(metrics, suffix="_mape")
    if primary is not None:
        model_name = primary.removesuffix("_mape")
        metrics["mape"] = metrics[f"{model_name}_mape"]
        metrics["mae"] = metrics.get(f"{model_name}_mae", 0.0)
        metrics["rmse"] = metrics.get(f"{model_name}_rmse", 0.0)
        metrics["r2"] = metrics.get(f"{model_name}_r2", 0.0)
        metrics["primary_model"] = model_name
    return metrics


def select_primary_metric(metrics: dict[str, Any], *, suffix: str) -> str | None:
    """Select the lowest numeric metric with the requested suffix."""

    candidates = {
        key: float(value)
        for key, value in metrics.items()
        if key.endswith(suffix) and isinstance(value, (int, float, np.floating))
    }
    if not candidates:
        return None
    return min(candidates, key=candidates.get)


def train_forecast_model(
    train_df: pd.DataFrame,
    config: ForecastTrainingConfig,
    *,
    hist_exog: list[str] | None = None,
):
    """Run CV, train the full forecast model, and return model plus metrics."""

    set_forecast_seed(config.seed)
    nf = build_forecast_engine(config, hist_exog=hist_exog)
    cv_df = nf.cross_validation(
        df=train_df,
        n_windows=config.cv_windows,
        step_size=config.horizon,
    )
    metrics = calculate_forecast_metrics(cv_df, eval_horizon=config.eval_horizon)
    nf.fit(df=train_df)
    return nf, cv_df, metrics


def save_forecast_model(model, model_dir: str | os.PathLike[str]) -> str:
    """Save a NeuralForecast model checkpoint directory."""

    path = Path(model_dir)
    if path.exists():
        shutil.rmtree(path)
    model.save(path=str(path), model_index=None, overwrite=True, save_dataset=False)
    return str(path)


def archive_model_dir(
    model_dir: str | os.PathLike[str],
    archive_path: str | os.PathLike[str],
) -> str:
    """Zip a checkpoint directory so ClearML can store it as model weights."""

    model_dir = Path(model_dir)
    archive_path = Path(archive_path)
    if archive_path.exists():
        archive_path.unlink()

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(model_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(model_dir))
    return str(archive_path)


def load_forecast_model_from_archive(archive_path: str | os.PathLike[str]):
    """Load a NeuralForecast model from a zipped checkpoint artifact."""

    from neuralforecast import NeuralForecast

    temp_dir = tempfile.mkdtemp(prefix="deposit_forecast_model_")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(temp_dir)
    return NeuralForecast.load(path=temp_dir, weights_only=False)
