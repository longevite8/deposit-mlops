"""
LightGBM Model Implementation - Gradient Boosting Regression.
Sử dụng LightGBM cho tabular time series forecasting.
"""

import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
import joblib

from business.models.base import (
    HyperparameterOptimizer,
    ModelTrainer,
    FeatureImportanceCalculator,
    ModelConfig,
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

    # Training
    early_stopping_rounds: int = 50
    verbose: int = -1
    metric: str = "mape"


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

        # Train model
        model = LGBMRegressor(**params)
        model.fit(self.X_train, self.y_train)

        # Evaluate on validation set
        y_pred = model.predict(self.X_valid)
        mape = mean_absolute_percentage_error(self.y_valid, y_pred)

        return mape

    def get_search_space(self) -> Dict[str, Any]:
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
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        best_params: Optional[Dict] = None,
        callbacks: Optional[list] = None,
    ) -> LGBMRegressor:
        """
        Train LightGBM model.

        Args:
            X_train, y_train: Training data
            X_valid, y_valid: Validation data
            best_params: Dict of best hyperparameters từ HPO
            callbacks: Optional callbacks

        Returns:
            Trained LGBMRegressor model
        """
        if best_params is None:
            best_params = {}

        # Merge best_params with defaults
        model_params = {
            "n_estimators": best_params.get("n_estimators", 200),
            "learning_rate": best_params.get("learning_rate", 0.05),
            "num_leaves": best_params.get("num_leaves", 31),
            "max_depth": best_params.get("max_depth", 10),
            "random_state": self.config.random_state,
            "verbose": self.config.verbose,
        }

        # Create model
        model = LGBMRegressor(**model_params)

        # Prepare eval_set
        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid, y_valid)]

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

        Args:
            X: Input features

        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)


class LGBMImportanceCalculator(FeatureImportanceCalculator):
    """
    Calculate feature importance từ LightGBM model.
    """

    def calculate_importance(
        self, model: LGBMRegressor, feature_names: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
