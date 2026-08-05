"""
NBEATSx Model Implementation - Dùng NeuralForecast.
Requires: pip install neuralforecast
"""

import warnings
from typing import Any

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")

from business.models.base import (
    HyperparameterOptimizer,
    ModelConfig,
    ModelTrainer,
)

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
except ImportError:
    raise ImportError(
        "NBEATSx requires: pip install neuralforecast. "
        "Please install before using NBEATSx model."
    )


class NBEATSxConfig(ModelConfig):
    """Configuration cho NBEATSx model."""

    # Model architecture
    input_size: int = 8
    forecast_horizon: int = 1  # Will be overridden by config value

    # Training hyperparameters
    max_steps: int = 100

    def __init__(self, random_state: int = 42, forecast_horizon: int = None, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        if forecast_horizon is not None:
            self.forecast_horizon = forecast_horizon


class NBEATSxOptimizer(HyperparameterOptimizer):
    """Hyperparameter Optimizer cho NBEATSx using Optuna."""

    def __init__(self, config: NBEATSxConfig):
        super().__init__(config)
        self.config: NBEATSxConfig = config

    def setup_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> None:
        """Setup dữ liệu cho NeuralForecast."""
        super().setup_data(X_train, y_train, X_valid, y_valid)

        from business.models.utils import normalize_data

        # Normalize data
        self.y_train_norm, self.scaler_y = normalize_data(y_train.values.reshape(-1, 1))
        self.y_valid_norm, _ = normalize_data(
            y_valid.values.reshape(-1, 1), self.scaler_y
        )

        # Prepare dataframes for NeuralForecast
        # NeuralForecast expects: ds, y, unique_id
        train_dates = pd.date_range(start="2020-01-01", periods=len(self.y_train_norm))
        valid_dates = pd.date_range(start="2020-01-01", periods=len(self.y_valid_norm))

        self.train_df = pd.DataFrame(
            {"ds": train_dates, "y": self.y_train_norm.flatten(), "unique_id": "target"}
        )

        self.valid_df = pd.DataFrame(
            {"ds": valid_dates, "y": self.y_valid_norm.flatten(), "unique_id": "target"}
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function cho Optuna - minimize validation loss."""
        try:
            from config import (
                HPO_INPUT_SIZE_OPTIONS,
                HPO_MAX_STEPS_MAX,
                HPO_MAX_STEPS_MIN,
                HPO_RANDOM_SEED_MAX,
                HPO_RANDOM_SEED_MIN,
            )

            # Suggest hyperparameters from config
            input_size = trial.suggest_categorical("input_size", HPO_INPUT_SIZE_OPTIONS)
            max_steps = trial.suggest_int(
                "max_steps", HPO_MAX_STEPS_MIN, HPO_MAX_STEPS_MAX
            )
            random_seed = trial.suggest_int(
                "random_seed", HPO_RANDOM_SEED_MIN, HPO_RANDOM_SEED_MAX
            )

            # Create NBEATSx model
            # Use stack_types=['identity'] to disable seasonality/trend (incompatible with h=1)
            model = NBEATSx(
                h=self.config.forecast_horizon,
                input_size=input_size,
                max_steps=max_steps,
                random_seed=random_seed,
                stack_types=["identity"],
                enable_progress_bar=False,
            )

            # Train NeuralForecast with validation data
            # NeuralForecast will compute validation loss internally during training
            nf = NeuralForecast(models=[model], freq="D")
            nf.fit(self.train_df, val_df=self.valid_df)

            # Extract validation loss that was computed during training
            from business.models.utils import compute_validation_loss_neural

            val_loss = compute_validation_loss_neural(
                model, self.valid_df, self.train_df
            )

            return val_loss

        except Exception as e:
            import traceback

            print(f"NBEATSx trial failed: {e!s}")
            print(traceback.format_exc())
            return float("inf")

    def get_search_space(self) -> dict[str, Any]:
        """Return search space description."""
        return {
            "input_size": "[8, 12, 16, 20, 24]",
            "max_steps": "[10, 20]",
            "random_seed": "[1, 10]",
        }


class NBEATSxTrainer(ModelTrainer):
    """Model Trainer cho NBEATSx."""

    def __init__(self, config: NBEATSxConfig):
        super().__init__(config)
        self.config: NBEATSxConfig = config
        self.scaler_y = None
        self.nf = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        best_params: dict | None = None,
        callbacks: list | None = None,
    ) -> Any:
        """Train NBEATSx model."""
        if best_params is None:
            best_params = {}

        from business.models.utils import normalize_data

        if not isinstance(y_train, pd.Series):
            raise TypeError(
                "NBEATSx requires y_train to be a pandas Series "
                "containing the original target column."
            )

        if y_train.empty:
            raise ValueError("NBEATSx cannot be trained with an empty target series.")

        # Normalize the original target series.
        y_train_norm, self.scaler_y = normalize_data(y_train.to_numpy().reshape(-1, 1))

        # NeuralForecast expects: unique_id, ds, y.
        train_dates = pd.date_range(
            start="2020-01-01",
            periods=len(y_train_norm),
            freq="D",
        )

        train_df = pd.DataFrame(
            {
                "unique_id": "target",
                "ds": train_dates,
                "y": y_train_norm.ravel(),
            }
        )

        model = NBEATSx(
            h=self.config.forecast_horizon,
            input_size=best_params.get(
                "input_size",
                self.config.input_size,
            ),
            max_steps=best_params.get(
                "max_steps",
                self.config.max_steps,
            ),
            random_seed=best_params.get(
                "random_seed",
                self.config.random_state,
            ),
            stack_types=["identity"],
            enable_progress_bar=False,
        )

        self.nf = NeuralForecast(
            models=[model],
            freq="D",
        )

        self.nf.fit(train_df)

        self.model = model
        self.feature_names = [y_train.name or "target"]

        return self.nf

    def predict(self) -> np.ndarray:
        """Forecast the next configured horizon."""
        if self.nf is None:
            raise ValueError("Model not trained yet. Call train() first.")

        if self.scaler_y is None:
            raise ValueError("Target scaler is not available.")

        from business.models.utils import inverse_normalize

        forecasts = self.nf.predict()

        if "NBEATSx" in forecasts.columns:
            prediction_column = "NBEATSx"
        else:
            prediction_columns = [
                column
                for column in forecasts.columns
                if column not in ["unique_id", "ds"]
            ]

            if not prediction_columns:
                raise ValueError(
                    "Cannot find NBEATSx prediction column. "
                    f"Available columns: {forecasts.columns.tolist()}"
                )

            prediction_column = prediction_columns[0]

        y_pred_normalized = forecasts[prediction_column].to_numpy()

        y_pred = inverse_normalize(
            y_pred_normalized.reshape(-1, 1),
            self.scaler_y,
        ).ravel()

        expected_horizon = self.config.forecast_horizon

        if len(y_pred) != expected_horizon:
            raise ValueError(
                f"Expected {expected_horizon} forecasts, but received {len(y_pred)}."
            )

        return y_pred
