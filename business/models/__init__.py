"""
Model Registry and Factory Pattern Implementation.
Provides centralized access to all supported model implementations.
"""

from business.models.base import (
    FeatureImportanceCalculator,
    HyperparameterOptimizer,
    ModelConfig,
    ModelTrainer,
)
from business.models.lightgbm_model import (
    LGBMConfig,
    LGBMImportanceCalculator,
    LGBMOptimizer,
    LGBMTrainer,
)
from business.models.model_registry import (
    SUPPORTED_MODELS,
    get_importance_calculator,
    get_model_config_class,
    get_optimizer_class,
    get_trainer_class,
)
from business.models.nbeatsx_model import (
    NBEATSxConfig,
    NBEATSxOptimizer,
    NBEATSxTrainer,
)
from business.models.nhits_model import (
    NHITSConfig,
    NHITSOptimizer,
    NHITSTrainer,
)

__all__ = [
    # Base classes
    "HyperparameterOptimizer",
    "ModelTrainer",
    "FeatureImportanceCalculator",
    "ModelConfig",
    # LightGBM
    "LGBMOptimizer",
    "LGBMTrainer",
    "LGBMImportanceCalculator",
    "LGBMConfig",
    # NBEATSx
    "NBEATSxOptimizer",
    "NBEATSxTrainer",
    "NBEATSxConfig",
    # NHITS
    "NHITSOptimizer",
    "NHITSTrainer",
    "NHITSConfig",
    # Registry
    "get_optimizer_class",
    "get_trainer_class",
    "get_importance_calculator",
    "SUPPORTED_MODELS",
]
