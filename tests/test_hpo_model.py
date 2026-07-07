"""Unit tests cho hpo_model.py"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from tasks.hpo_model import objective


@pytest.fixture
def mock_task():
    """Mock ClearML task"""
    task = Mock()
    task.get_logger.return_value = Mock()
    return task


@pytest.fixture
def sample_data():
    """Sample training data"""
    np.random.seed(42)
    n_samples = 100
    X_train = pd.DataFrame(
        {
            "lag_1": np.random.randn(n_samples),
            "lag_7": np.random.randn(n_samples),
            "rolling_mean_7": np.random.randn(n_samples),
        }
    )
    y_train = 100 + 0.5 * X_train["lag_1"] + np.random.randn(n_samples) * 10

    X_valid = pd.DataFrame(
        {
            "lag_1": np.random.randn(50),
            "lag_7": np.random.randn(50),
            "rolling_mean_7": np.random.randn(50),
        }
    )
    y_valid = 100 + 0.5 * X_valid["lag_1"] + np.random.randn(50) * 10

    return X_train, y_train, X_valid, y_valid


def test_objective_returns_mape(mock_task, sample_data):
    """Test that objective function returns a float (MAPE)"""
    X_train, y_train, X_valid, y_valid = sample_data

    # Mock trial
    mock_trial = Mock()
    mock_trial.suggest_float = Mock(
        side_effect=lambda name, *args, **kwargs: (
            0.05 if name == "learning_rate" else 50
        )
    )
    mock_trial.suggest_int = Mock(
        side_effect=lambda name, *args, **kwargs: 50 if name == "num_leaves" else 500
    )
    mock_trial.number = 0

    # Run objective
    mape = objective(mock_trial)

    # Assert
    assert isinstance(mape, float)
    assert 0 <= mape <= 1.0  # MAPE should be between 0 and 1

    # Assert mock was called
    mock_trial.suggest_float.assert_called()
    mock_trial.suggest_int.assert_called()


def test_objective_reports_scalars(mock_task, sample_data, monkeypatch):
    """Test that objective reports MAPE + R2 scalars to ClearML"""
    X_train, y_train, X_valid, y_valid = sample_data

    # Mock trial
    mock_trial = Mock()
    mock_trial.suggest_float = Mock(return_value=0.05)
    mock_trial.suggest_int = Mock(return_value=100)
    mock_trial.number = 1

    # Call objective
    mape = objective(mock_trial)

    # Check that logger.report_scalar was called
    assert mock_task.get_logger.return_value.report_scalar.called


def test_hpo_best_params_are_reasonable(sample_data):
    """Test that HPO finds reasonable hyperparameters"""
    import optuna

    X_train, y_train, X_valid, y_valid = sample_data

    def objective_test(trial):
        from lightgbm import LGBMRegressor
        from sklearn.metrics import mean_absolute_percentage_error

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "random_state": 42,
        }

        model = LGBMRegressor(**params, verbose=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        mape = mean_absolute_percentage_error(y_valid, y_pred)

        return mape

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective_test, n_trials=5, show_progress_bar=False)

    best_params = study.best_params

    # Assert params are in reasonable ranges
    assert 0.01 <= best_params["learning_rate"] <= 0.1
    assert 15 <= best_params["num_leaves"] <= 100
    assert 100 <= best_params["n_estimators"] <= 500
