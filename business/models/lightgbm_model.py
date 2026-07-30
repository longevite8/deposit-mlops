"""
LightGBM Model Implementation - Gradient Boosting Regression.
Sử dụng LightGBM cho tabular time series forecasting.
"""

from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error

from business.models.base import (
    FeatureImportanceCalculator,
    HyperparameterOptimizer,
    ModelConfig,
    ModelTrainer,
)


class LGBMConfig(ModelConfig):
    """Configuration cho LightGBM model."""

    # HPO search space
    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.1
    num_leaves_min: int = 15
    num_leaves_max: int = 100
    n_estimators_min: int = 100
    n_estimators_max: int = 1000
    max_depth_min: int = 5
    max_depth_max: int = 15

    # Forecast horizon
    forecast_horizon: int = 1  # Will be overridden by config value

    # Training
    early_stopping_rounds: int = 50
    verbose: int = -1
    metric: str = "mape"

    def __init__(self, random_state: int = 42, forecast_horizon: int = None, **kwargs):
        super().__init__(random_state=random_state, **kwargs)
        if forecast_horizon is not None:
            self.forecast_horizon = forecast_horizon


class LGBMOptimizer(HyperparameterOptimizer):
    """
    Hyperparameter Optimizer cho LightGBM using Optuna.
    """

    def __init__(self, config: LGBMConfig):
        super().__init__(config)
        self.config: LGBMConfig = config

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function cho Optuna trial - tối thiểu MAPE.

        Multi-target strategy:
        - Trains on all target_1, target_2, ..., target_h
        - Evaluates on target_h (forecast_horizon step)

        Args:
            trial: Optuna trial object

        Returns:
            float: MAPE value (lower is better)
        """
        # Define hyperparameter search space
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.config.learning_rate_min,
                self.config.learning_rate_max,
                log=True,
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", self.config.num_leaves_min, self.config.num_leaves_max
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators",
                self.config.n_estimators_min,
                self.config.n_estimators_max,
            ),
            "max_depth": trial.suggest_int(
                "max_depth", self.config.max_depth_min, self.config.max_depth_max
            ),
            "random_state": self.config.random_state,
            "verbose": -1,
        }

        # Train model on all targets
        model = LGBMRegressor(**params)
        model.fit(self.X_train, self.y_train)

        # Evaluate on validation set
        # y_pred shape: (n_samples, n_targets) for multi-output
        # Extract prediction for target_h (forecast_horizon)
        y_pred = model.predict(self.X_valid)

        # For multi-target: y_pred has shape (n_samples, forecast_horizon)
        # Use last column (target_h)
        if len(y_pred.shape) > 1:
            y_pred_h = y_pred[:, -1]  # Last column = target_forecast_horizon
        else:
            y_pred_h = y_pred  # Single target fallback

        # y_valid is a DataFrame with multiple target columns
        # Extract target_h for evaluation
        if isinstance(self.y_valid, pd.DataFrame):
            y_valid_h = self.y_valid.iloc[:, -1].values  # Last column
        else:
            y_valid_h = self.y_valid  # Fallback

        mape = mean_absolute_percentage_error(y_valid_h, y_pred_h)

        return mape

    def get_search_space(self) -> dict[str, Any]:
        """
        Return search space description (used for logging).

        Returns:
            Dict describing the search space
        """
        return {
            "learning_rate": f"[{self.config.learning_rate_min}, {self.config.learning_rate_max}]",
            "num_leaves": f"[{self.config.num_leaves_min}, {self.config.num_leaves_max}]",
            "n_estimators": f"[{self.config.n_estimators_min}, {self.config.n_estimators_max}]",
            "max_depth": f"[{self.config.max_depth_min}, {self.config.max_depth_max}]",
        }


class LGBMTrainer(ModelTrainer):
    """
    Model Trainer cho LightGBM.
    """

    def __init__(self, config: LGBMConfig):
        super().__init__(config)
        self.config: LGBMConfig = config

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | pd.DataFrame | None = None,
        best_params: dict | None = None,
        callbacks: list | None = None,
    ) -> LGBMRegressor:
        """
        Train LightGBM model with multi-target strategy.

        Multi-target approach:
        - y_train/y_valid: DataFrame with columns target_1, target_2, ..., target_h
        - Trains on all targets simultaneously
        - Evaluates on target_h (forecast_horizon)

        Args:
            X_train, y_train: Training data (y_train can be DataFrame with multiple targets)
            X_valid, y_valid: Validation data (y_valid can be DataFrame with multiple targets)
            best_params: Dict of best hyperparameters từ HPO
            callbacks: Optional callbacks

        Returns:
            Trained LGBMRegressor model (multi-output)
        """
        if best_params is None:
            best_params = {}

        # Filter out verbose if it's a boolean (LightGBM 4.7+ requires int)
        if "verbose" in best_params and isinstance(best_params["verbose"], bool):
            del best_params["verbose"]

        # Merge best_params with defaults
        model_params = {
            "n_estimators": best_params.get("n_estimators", 200),
            "learning_rate": best_params.get("learning_rate", 0.05),
            "num_leaves": best_params.get("num_leaves", 31),
            "max_depth": best_params.get("max_depth", 10),
            "random_state": self.config.random_state,
            "verbose": -1,  # Ensure verbose is int, not boolean
        }

        # Create model
        model = LGBMRegressor(**model_params)

        # Prepare eval_set with target_h (forecast_horizon)
        eval_set = None
        if X_valid is not None and y_valid is not None:
            # Extract target_h for evaluation (last column in multi-target DataFrame)
            if isinstance(y_valid, pd.DataFrame):
                y_valid_h = y_valid.iloc[:, -1]  # Last column = target_forecast_horizon
            else:
                y_valid_h = y_valid
            eval_set = [(X_valid, y_valid_h)]

        # Train with optional callbacks
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        self.model = model
        self.feature_names = X_train.columns.tolist()
        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Multi-target approach:
        - Model predicts all targets (target_1, ..., target_h)
        - Returns only target_h (the forecast_horizon step)

        Args:
            X: Input features

        Returns:
            np.ndarray: Predictions for target_h (shape: (n_samples,))
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        y_pred = self.model.predict(X)

        # Extract prediction for target_h (last column for multi-output)
        if len(y_pred.shape) > 1:
            # Multi-output: shape (n_samples, forecast_horizon)
            # Return last column (target_h)
            return y_pred[:, -1]
        else:
            # Single output: shape (n_samples,)
            return y_pred


class LGBMImportanceCalculator(FeatureImportanceCalculator):
    """
    Calculate feature importance từ LightGBM model.
    """

    def calculate_importance(
        self, model: LGBMRegressor, feature_names: list
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate Split và Gain importance.

        Args:
            model: Trained LGBMRegressor
            feature_names: List of feature names

        Returns:
            Tuple of (split_importance_df, gain_importance_df)
        """
        # Split importance
        split_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "split_importance": model.booster_.feature_importance(
                    importance_type="split"
                ),
            }
        ).sort_values("split_importance", ascending=False)

        # Gain importance
        gain_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "gain_importance": model.booster_.feature_importance(
                    importance_type="gain"
                ),
            }
        ).sort_values("gain_importance", ascending=False)

        return split_importance, gain_importance


def save_lightgbm_model(
    model: LGBMRegressor, filepath: str = "lightgbm_model.pkl"
) -> str:
    """
    Save LightGBM model to file.

    Args:
        model: Trained LGBMRegressor
        filepath: Path to save model

    Returns:
        Path to saved model
    """
    joblib.dump(model, filepath)
    return filepath
