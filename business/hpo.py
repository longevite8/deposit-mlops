"""
Generic Hyperparameter Optimization Orchestrator.
Hoạt động với bất kỳ model type nào thông qua Strategy pattern.
"""

import optuna
from typing import Optional, Any
import pandas as pd


def run_generic_hpo_optimization(
    optimizer: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int = 50,
    random_state: int = 42,
    callbacks: Optional[list] = None,
) -> optuna.Study:
    """
    Thực hiện tối ưu hóa Hyperparameters bằng Optuna cho bất kỳ model type nào.

    Hàm này generic và không phụ thuộc vào implementation chi tiết của từng model.
    Nó chỉ yêu cầu:
    1. Optimizer object có method: objective(trial) và setup_data()
    2. Optuna framework để manage trials

    Args:
        optimizer: HyperparameterOptimizer instance (e.g., LGBMOptimizer, NBEATSxOptimizer)
        X_train: Training features (DataFrame)
        y_train: Training target (Series)
        X_valid: Validation features (DataFrame)
        y_valid: Validation target (Series)
        n_trials: Số trials cần chạy (default: 50)
        random_state: Random seed cho reproducibility
        callbacks: Optional list of callbacks để execute sau mỗi trial

    Returns:
        optuna.Study object chứa kết quả HPO

    Example:
        >>> from business.models import get_optimizer_class, LGBMConfig
        >>> optimizer_class = get_optimizer_class("lightgbm")
        >>> optimizer = optimizer_class(config=LGBMConfig())
        >>> study = run_generic_hpo_optimization(
        ...     optimizer=optimizer,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_valid=X_valid,
        ...     y_valid=y_valid,
        ...     n_trials=50,
        ... )
    """

    # =====================================================
    # Setup: Prepare data cho optimizer
    # =====================================================

    optimizer.setup_data(X_train, y_train, X_valid, y_valid)

    # =====================================================
    # Create Study: Initialize Optuna study
    # =====================================================

    study = optuna.create_study(
        direction="minimize",  # Minimize MAPE/loss
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    # =====================================================
    # Optimize: Run trials
    # =====================================================

    study.optimize(
        objective=optimizer.objective,
        n_trials=n_trials,
        callbacks=callbacks,
    )

    # =====================================================
    # Return Study
    # =====================================================

    return study
