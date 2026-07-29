"""
NHITS Model Implementation - Dùng NeuralForecast.
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
    from neuralforecast.models import NHITS
except ImportError:
    raise ImportError(
        "NHITS requires: pip install neuralforecast. "
        "Please install before using NHITS model."
    )


class NHITSConfig(ModelConfig):
    """Configuration cho NHITS model."""

    # HPO search space
    n_layers_min: int = 2
    n_layers_max: int = 5
    n_hidden_min: int = 64
    n_hidden_max: int = 256
    dropout_min: float = 0.0
    dropout_max: float = 0.3

    # Model architecture
    input_size: int = 8
    forecast_horizon: int = 1


class NHITSOptimizer(HyperparameterOptimizer):
    """Hyperparameter Optimizer cho NHITS using Optuna."""

    def __init__(self, config: NHITSConfig):
        super().__init__(config)
        self.config: NHITSConfig = config

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
        """Objective function cho Optuna - minimize MAPE."""
        try:
            # Suggest hyperparameters
            n_layers = trial.suggest_int(
                "n_layers", self.config.n_layers_min, self.config.n_layers_max
            )
            n_hidden = trial.suggest_int(
                "n_hidden", self.config.n_hidden_min, self.config.n_hidden_max
            )
            dropout = trial.suggest_float(
                "dropout", self.config.dropout_min, self.config.dropout_max
            )

            # Create NHITS model with architecture params only
            model = NHITS(
                h=self.config.forecast_horizon,
                input_size=self.config.input_size,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout=dropout,
                random_seed=self.config.random_state,
            )

            # Train NeuralForecast with validation data for early stopping
            nf = NeuralForecast(models=[model], freq="D")
            nf.fit(self.train_df, val_df=self.valid_df)

            # Validate
            forecasts = nf.predict(self.valid_df)
            y_pred = forecasts["NHITS"].values

            # Inverse normalize
            from business.models.utils import inverse_normalize

            y_pred = inverse_normalize(y_pred.reshape(-1, 1), self.scaler_y).flatten()
            y_valid = inverse_normalize(self.y_valid_norm, self.scaler_y).flatten()

            # Calculate MAPE
            from sklearn.metrics import mean_absolute_percentage_error

            mape = mean_absolute_percentage_error(y_valid, y_pred)
            return mape

        except Exception as e:
            print(f"NHITS trial failed: {e!s}")
            return float("inf")

    def get_search_space(self) -> dict[str, Any]:
        """Return search space description."""
        return {
            "n_layers": f"[{self.config.n_layers_min}, {self.config.n_layers_max}]",
            "n_hidden": f"[{self.config.n_hidden_min}, {self.config.n_hidden_max}]",
            "dropout": f"[{self.config.dropout_min}, {self.config.dropout_max}]",
        }


class NHITSTrainer(ModelTrainer):
    """Model Trainer cho NHITS."""

    def __init__(self, config: NHITSConfig):
        super().__init__(config)
        self.config: NHITSConfig = config
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
        """Train NHITS model."""
        if best_params is None:
            best_params = {}

        from business.models.utils import normalize_data

        # Normalize data
        y_train_norm, self.scaler_y = normalize_data(y_train.values.reshape(-1, 1))

        # Prepare dataframe for NeuralForecast
        train_dates = pd.date_range(start="2020-01-01", periods=len(y_train_norm))
        train_df = pd.DataFrame(
            {"ds": train_dates, "y": y_train_norm.flatten(), "unique_id": "target"}
        )

        # Create NHITS model with architecture params only
        model = NHITS(
            h=self.config.forecast_horizon,
            input_size=self.config.input_size,
            n_layers=best_params.get("n_layers", 3),
            n_hidden=best_params.get("n_hidden", 128),
            dropout=best_params.get("dropout", 0.1),
            random_seed=self.config.random_state,
        )

        # Train with NeuralForecast
        self.nf = NeuralForecast(models=[model], freq="D")
        self.nf.fit(train_df)

        self.model = model
        self.feature_names = y_train.name if hasattr(y_train, "name") else ["target"]
        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.nf is None:
            raise ValueError("Model not trained yet. Call train() first.")

        from business.models.utils import inverse_normalize

        # Create dataframe for prediction
        pred_dates = pd.date_range(start="2020-01-01", periods=len(X))
        pred_df = pd.DataFrame({"ds": pred_dates, "unique_id": "target"})

        # Predict
        forecasts = self.nf.predict(pred_df)
        y_pred = forecasts["NHITS"].values

        # Inverse normalize
        y_pred = inverse_normalize(y_pred.reshape(-1, 1), self.scaler_y).flatten()

        return y_pred
