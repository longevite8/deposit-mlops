"""
Generic Model Training Orchestrator.
Hoạt động với bất kỳ model type nào thông qua Strategy pattern.
"""

import pandas as pd
from typing import Optional, Any, Tuple
import joblib


def train_model(
    trainer: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: Optional[pd.DataFrame] = None,
    y_valid: Optional[pd.Series] = None,
    best_params: Optional[dict] = None,
    callbacks: Optional[list] = None,
) -> Any:
    """
    Huấn luyện mô hình với bất kỳ model type nào.

    Hàm này generic và không phụ thuộc vào implementation chi tiết của từng model.
    Nó chỉ yêu cầu trainer object có method: train(X_train, y_train, ...)

    Args:
        trainer: ModelTrainer instance (e.g., LGBMTrainer, NBEATSxTrainer)
        X_train: Training features (DataFrame)
        y_train: Training target (Series)
        X_valid: Validation features (optional)
        y_valid: Validation target (optional)
        best_params: Dict of best hyperparameters từ HPO
        callbacks: Optional callbacks

    Returns:
        Trained model object

    Example:
        >>> from business.models import get_trainer_class, LGBMConfig
        >>> trainer_class = get_trainer_class("lightgbm")
        >>> trainer = trainer_class(config=LGBMConfig())
        >>> model = train_model(
        ...     trainer=trainer,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_valid=X_valid,
        ...     y_valid=y_valid,
        ...     best_params=best_params,
        ... )
    """

    # =====================================================
    # Train Model
    # =====================================================

    model = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        best_params=best_params,
        callbacks=callbacks,
    )

    # =====================================================
    # Return Model
    # =====================================================

    return model


def calculate_feature_importance(
    calculator: Any,
    model: Any,
    feature_names: list,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Tính toán Feature Importance từ trained model (nếu supported).

    Một số models (e.g., neural networks) không hỗ trợ feature importance.
    Trong trường hợp đó, calculator sẽ là None và function này không được gọi.

    Args:
        calculator: FeatureImportanceCalculator instance hoặc None
        model: Trained model object
        feature_names: List of feature column names

    Returns:
        Tuple of (primary_importance, secondary_importance)
        - primary_importance: Main importance scores (e.g., Split importance)
        - secondary_importance: Alternative scores (e.g., Gain importance) hoặc None

    Raises:
        ValueError: Nếu calculator là None

    Example:
        >>> from business.models import get_importance_calculator, LGBMImportanceCalculator
        >>> calc = LGBMImportanceCalculator()
        >>> split_imp, gain_imp = calculate_feature_importance(calc, model, feature_names)
    """

    if calculator is None:
        raise ValueError(
            "Feature importance calculator is None. Model type không hỗ trợ feature importance."
        )

    # =====================================================
    # Calculate Importance
    # =====================================================

    primary_importance, secondary_importance = calculator.calculate_importance(
        model=model,
        feature_names=feature_names,
    )

    # =====================================================
    # Return
    # =====================================================

    return primary_importance, secondary_importance


def save_model(model: Any, filepath: str = "model.pkl") -> str:
    """
    Lưu trained model vào file.

    Args:
        model: Trained model object
        filepath: Path để lưu model

    Returns:
        Path đến file model
    """
    joblib.dump(model, filepath)
    return filepath
