"""
Abstract Base Classes cho Model Implementations.
Định nghĩa interface chung cho tất cả các thuật toán.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class ModelConfig:
    """
    Base configuration class cho model.
    Các implementation cụ thể kế thừa và mở rộng class này.
    """

    random_state: int = 42
    verbose: bool = False


class HyperparameterOptimizer(ABC):
    """
    Abstract base class cho HPO implementations.

    Mỗi model type phải implement:
    - objective(): hàm mục tiêu cho Optuna trial
    - get_search_space(): định nghĩa search space của parameters
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize optimizer with configuration.

        Args:
            config: ModelConfig instance chứa các tham số chung
        """
        self.config = config

    @abstractmethod
    def objective(self, trial) -> float:
        """
        Objective function cho Optuna trial.
        Phải return: metric value (float) để optimize.

        Args:
            trial: Optuna trial object

        Returns:
            float: Metric value (e.g., MAPE, loss)
        """
        pass

    @abstractmethod
    def get_search_space(self) -> Dict[str, Any]:
        """
        Define hyperparameter search space.
        Return dict chứa trial.suggest_* calls cho mỗi parameter.

        Returns:
            Dict: Mapping of parameter names to suggested values
        """
        pass

    def setup_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> None:
        """
        Setup training and validation data.
        Mỗi implementation có thể override để prepare data đặc thù.

        Args:
            X_train, y_train: Training data
            X_valid, y_valid: Validation data
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.feature_names = X_train.columns.tolist()


class ModelTrainer(ABC):
    """
    Abstract base class cho training implementations.

    Mỗi model type phải implement:
    - train(): huấn luyện mô hình
    - predict(): dự đoán trên data mới
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: ModelConfig instance chứa các tham số chung
        """
        self.config = config
        self.model = None

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        best_params: Optional[Dict] = None,
        callbacks: Optional[list] = None,
    ) -> Any:
        """
        Train model with given parameters.

        Args:
            X_train, y_train: Training data
            X_valid, y_valid: Validation data (optional)
            best_params: Dict of hyperparameters
            callbacks: Optional list of callbacks

        Returns:
            Trained model object
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input features

        Returns:
            np.ndarray: Predictions
        """
        pass

    def get_model(self) -> Any:
        """
        Return trained model object.

        Returns:
            Trained model instance
        """
        return self.model


class FeatureImportanceCalculator(ABC):
    """
    Abstract base class cho Feature Importance calculations.
    Không phải tất cả models đều hỗ trợ feature importance (e.g., neural models).
    """

    @abstractmethod
    def calculate_importance(
        self, model: Any, feature_names: list
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Calculate feature importance from trained model.

        Args:
            model: Trained model object
            feature_names: List of feature column names

        Returns:
            Tuple of (primary_importance_df, optional_secondary_importance_df)
            - primary_importance_df: Main importance scores (e.g., Split importance)
            - optional_secondary_importance_df: Alternative scores or None (e.g., Gain importance)
        """
        pass
