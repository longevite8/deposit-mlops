"""
NBEATSx Model Implementation - Neural Forecasting.
Sử dụng BATS/NBEATS với exogenous variables cho time series forecasting.
Requires: pip install darts torch pytorch-lightning
"""

import optuna
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
import warnings

warnings.filterwarnings("ignore")

from business.models.base import (
    HyperparameterOptimizer,
    ModelTrainer,
    ModelConfig,
)

try:
    from darts import TimeSeries
    from darts.models import NBEATSModel
    from darts.metrics import mape as darts_mape
    from pytorch_lightning.callbacks import EarlyStopping
except ImportError:
    raise ImportError(
        "NBEATSx requires: pip install darts torch pytorch-lightning. "
        "Please install before using NBEATSx model."
    )


class NBEATSxConfig(ModelConfig):
    """Configuration cho NBEATSx model."""

    # HPO search space
    stack_sizes_min: int = 4
    stack_sizes_max: int = 16
    hidden_size_min: int = 32
    hidden_size_max: int = 256
    num_stacks: int = 2
    num_layers_min: int = 2
    num_layers_max: int = 5

    # Training
    input_size: int = 8
    forecast_horizon: int = 1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10


class NBEATSxOptimizer(HyperparameterOptimizer):
    """
    Hyperparameter Optimizer cho NBEATSx using Optuna.
    """

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
        """
        Setup và prepare dữ liệu cho neural models.
        """
        super().setup_data(X_train, y_train, X_valid, y_valid)

        from business.models.utils import create_temporal_tensors, normalize_data

        # Normalize data
        self.X_train_norm, self.scaler_X = normalize_data(X_train)
        self.y_train_norm, self.scaler_y = normalize_data(y_train.values.reshape(-1, 1))
        self.X_valid_norm, _ = normalize_data(X_valid, self.scaler_X)
        self.y_valid_norm, _ = normalize_data(
            y_valid.values.reshape(-1, 1), self.scaler_y
        )

        # Create temporal tensors
        self.X_train_ts, self.y_train_ts = create_temporal_tensors(
            self.X_train_norm,
            self.y_train_norm.flatten(),
            input_size=self.config.input_size,
            horizon=self.config.forecast_horizon,
        )
        self.X_valid_ts, self.y_valid_ts = create_temporal_tensors(
            self.X_valid_norm,
            self.y_valid_norm.flatten(),
            input_size=self.config.input_size,
            horizon=self.config.forecast_horizon,
        )

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function cho Optuna - tối thiểu MAPE.
        """
        try:
            # Suggest hyperparameters
            stack_size = trial.suggest_int(
                "stack_size", self.config.stack_sizes_min, self.config.stack_sizes_max
            )
            hidden_size = trial.suggest_int(
                "hidden_size", self.config.hidden_size_min, self.config.hidden_size_max
            )
            num_layers = trial.suggest_int(
                "num_layers", self.config.num_layers_min, self.config.num_layers_max
            )
            dropout = trial.suggest_float("dropout", 0.0, 0.3)

            # Create NBEATSx model
            model = NBEATSModel(
                input_chunk_length=self.config.input_size,
                output_chunk_length=self.config.forecast_horizon,
                num_stacks=self.config.num_stacks,
                num_blocks=1,
                num_layers=num_layers,
                layer_widths=hidden_size,
                expansion_coefficient_dim=stack_size,
                trend_polynomial_degree=2,
                dropout=dropout,
                random_state=self.config.random_state,
            )

            # Train with early stopping
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                min_delta=1e-4,
                mode="min",
            )

            # Convert to TimeSeries
            ts_train = TimeSeries.from_values(self.X_train_ts)
            ts_valid = TimeSeries.from_values(self.X_valid_ts)
            y_ts_train = TimeSeries.from_values(self.y_train_ts)
            y_ts_valid = TimeSeries.from_values(self.y_valid_ts)

            # Train
            model.fit(
                series=y_ts_train,
                past_covariates=ts_train,
                val_series=y_ts_valid,
                val_past_covariates=ts_valid,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                verbose=False,
                callbacks=[early_stop],
            )

            # Evaluate
            y_pred_ts = model.predict(len(self.y_valid_ts), past_covariates=ts_valid)
            y_pred = y_pred_ts.values().flatten()

            # Inverse normalize
            from business.models.utils import inverse_normalize

            y_pred = inverse_normalize(y_pred.reshape(-1, 1), self.scaler_y).flatten()
            y_valid = inverse_normalize(self.y_valid_norm, self.scaler_y).flatten()

            # Calculate MAPE
            from sklearn.metrics import mean_absolute_percentage_error

            mape = mean_absolute_percentage_error(y_valid, y_pred)

            return mape

        except Exception as e:
            print(f"NBEATSx trial failed: {str(e)}")
            return float("inf")

    def get_search_space(self) -> Dict[str, Any]:
        """Return search space description."""
        return {
            "stack_size": f"[{self.config.stack_sizes_min}, {self.config.stack_sizes_max}]",
            "hidden_size": f"[{self.config.hidden_size_min}, {self.config.hidden_size_max}]",
            "num_layers": f"[{self.config.num_layers_min}, {self.config.num_layers_max}]",
            "dropout": "[0.0, 0.3]",
        }


class NBEATSxTrainer(ModelTrainer):
    """
    Model Trainer cho NBEATSx.
    """

    def __init__(self, config: NBEATSxConfig):
        super().__init__(config)
        self.config: NBEATSxConfig = config
        self.scaler_X = None
        self.scaler_y = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        best_params: Optional[Dict] = None,
        callbacks: Optional[list] = None,
    ) -> NBEATSModel:
        """
        Train NBEATSx model.
        """
        if best_params is None:
            best_params = {}

        from business.models.utils import create_temporal_tensors, normalize_data

        # Normalize data
        X_train_norm, self.scaler_X = normalize_data(X_train)
        y_train_norm, self.scaler_y = normalize_data(y_train.values.reshape(-1, 1))
        X_valid_norm, _ = (
            normalize_data(X_valid, self.scaler_X)
            if X_valid is not None
            else (None, None)
        )
        y_valid_norm, _ = (
            normalize_data(y_valid.values.reshape(-1, 1), self.scaler_y)
            if y_valid is not None
            else (None, None)
        )

        # Create temporal tensors
        X_train_ts, y_train_ts = create_temporal_tensors(
            X_train_norm,
            y_train_norm.flatten(),
            input_size=self.config.input_size,
            horizon=self.config.forecast_horizon,
        )

        X_valid_ts, y_valid_ts = None, None
        if X_valid is not None and y_valid is not None:
            X_valid_ts, y_valid_ts = create_temporal_tensors(
                X_valid_norm,
                y_valid_norm.flatten(),
                input_size=self.config.input_size,
                horizon=self.config.forecast_horizon,
            )

        # Create model
        model_params = {
            "input_chunk_length": self.config.input_size,
            "output_chunk_length": self.config.forecast_horizon,
            "num_stacks": self.config.num_stacks,
            "num_blocks": 1,
            "num_layers": best_params.get("num_layers", 3),
            "layer_widths": best_params.get("hidden_size", 128),
            "expansion_coefficient_dim": best_params.get("stack_size", 8),
            "trend_polynomial_degree": 2,
            "dropout": best_params.get("dropout", 0.1),
            "random_state": self.config.random_state,
        }

        model = NBEATSModel(**model_params)

        # Prepare training callbacks
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            min_delta=1e-4,
            mode="min",
        )

        train_callbacks = [early_stop]
        if callbacks:
            train_callbacks.extend(callbacks)

        # Convert to TimeSeries
        ts_train = TimeSeries.from_values(X_train_ts)
        y_ts_train = TimeSeries.from_values(y_train_ts)

        ts_valid, y_ts_valid = None, None
        if X_valid_ts is not None:
            ts_valid = TimeSeries.from_values(X_valid_ts)
            y_ts_valid = TimeSeries.from_values(y_valid_ts)

        # Train
        model.fit(
            series=y_ts_train,
            past_covariates=ts_train,
            val_series=y_ts_valid,
            val_past_covariates=ts_valid,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=False,
            callbacks=train_callbacks,
        )

        self.model = model
        self.feature_names = X_train.columns.tolist()
        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        from business.models.utils import (
            normalize_data,
            inverse_normalize,
            create_temporal_tensors,
        )

        # Normalize
        X_norm, _ = normalize_data(X, self.scaler_X)

        # Create temporal tensor (dummy y for shape)
        X_ts, _ = create_temporal_tensors(
            X_norm,
            np.zeros(len(X_norm)),
            input_size=self.config.input_size,
            horizon=self.config.forecast_horizon,
        )

        # Predict
        ts_input = TimeSeries.from_values(X_ts)
        y_pred_ts = self.model.predict(len(X_ts), past_covariates=ts_input)
        y_pred = y_pred_ts.values().flatten()

        # Inverse normalize
        y_pred = inverse_normalize(y_pred.reshape(-1, 1), self.scaler_y).flatten()

        return y_pred
