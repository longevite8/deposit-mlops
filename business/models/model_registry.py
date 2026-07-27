"""
Model Registry và Factory Pattern Implementation.
Cung cấp centralized access đến tất cả model implementations.
"""

from typing import Type, Optional
from business.models.base import (
    HyperparameterOptimizer,
    ModelTrainer,
    FeatureImportanceCalculator,
    ModelConfig,
)

# Define supported models
SUPPORTED_MODELS = ["lightgbm", "nbeatsx", "nhits"]


def get_optimizer_class(model_type: str) -> Type[HyperparameterOptimizer]:
    """
    Get Optimizer class dựa trên model type.

    Args:
        model_type: Tên model ("lightgbm", "nbeatsx", "nhits")

    Returns:
        Optimizer class

    Raises:
        ValueError: Nếu model type không được support
    """
    model_type = model_type.lower()

    if model_type == "lightgbm":
        from business.models.lightgbm_model import LGBMOptimizer

        return LGBMOptimizer

    elif model_type == "nbeatsx":
        from business.models.nbeatsx_model import NBEATSxOptimizer

        return NBEATSxOptimizer

    elif model_type == "nhits":
        from business.models.nhits_model import NHITSOptimizer

        return NHITSOptimizer

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported models: {SUPPORTED_MODELS}"
        )


def get_trainer_class(model_type: str) -> Type[ModelTrainer]:
    """
    Get Trainer class dựa trên model type.

    Args:
        model_type: Tên model ("lightgbm", "nbeatsx", "nhits")

    Returns:
        Trainer class

    Raises:
        ValueError: Nếu model type không được support
    """
    model_type = model_type.lower()

    if model_type == "lightgbm":
        from business.models.lightgbm_model import LGBMTrainer

        return LGBMTrainer

    elif model_type == "nbeatsx":
        from business.models.nbeatsx_model import NBEATSxTrainer

        return NBEATSxTrainer

    elif model_type == "nhits":
        from business.models.nhits_model import NHITSTrainer

        return NHITSTrainer

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported models: {SUPPORTED_MODELS}"
        )


def get_importance_calculator(
    model_type: str,
) -> Optional[Type[FeatureImportanceCalculator]]:
    """
    Get Feature Importance Calculator class (if available) dựa trên model type.

    Note: Một số models (neural models) không hỗ trợ feature importance.
    Trong trường hợp đó, return None.

    Args:
        model_type: Tên model ("lightgbm", "nbeatsx", "nhits")

    Returns:
        Importance calculator class hoặc None nếu không hỗ trợ

    Raises:
        ValueError: Nếu model type không được support
    """
    model_type = model_type.lower()

    if model_type == "lightgbm":
        from business.models.lightgbm_model import LGBMImportanceCalculator

        return LGBMImportanceCalculator

    elif model_type in ["nbeatsx", "nhits"]:
        # Neural models không hỗ trợ feature importance
        return None

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported models: {SUPPORTED_MODELS}"
        )


def get_model_config_class(model_type: str) -> Type[ModelConfig]:
    """
    Get Model Config class dựa trên model type.

    Args:
        model_type: Tên model ("lightgbm", "nbeatsx", "nhits")

    Returns:
        Config class

    Raises:
        ValueError: Nếu model type không được support
    """
    model_type = model_type.lower()

    if model_type == "lightgbm":
        from business.models.lightgbm_model import LGBMConfig

        return LGBMConfig

    elif model_type == "nbeatsx":
        from business.models.nbeatsx_model import NBEATSxConfig

        return NBEATSxConfig

    elif model_type == "nhits":
        from business.models.nhits_model import NHITSConfig

        return NHITSConfig

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported models: {SUPPORTED_MODELS}"
        )


def validate_model_type(model_type: str) -> bool:
    """
    Validate if model type is supported.

    Args:
        model_type: Tên model để validate

    Returns:
        True nếu model type được support, False nếu không
    """
    return model_type.lower() in SUPPORTED_MODELS
