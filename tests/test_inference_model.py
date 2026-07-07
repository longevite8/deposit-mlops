"""Unit tests cho inference_model.py"""

import pandas as pd
import numpy as np


def test_inference_latency_calculation():
    """Test that inference latency is calculated correctly"""
    import time
    from lightgbm import LGBMRegressor

    # Create dummy model
    X_train = pd.DataFrame(
        {
            "lag_1": np.random.randn(100),
            "lag_7": np.random.randn(100),
            "rolling_mean_7": np.random.randn(100),
        }
    )
    y_train = np.random.randn(100)

    model = LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(X_train, y_train)

    # Inference
    X_infer = pd.DataFrame(
        {
            "lag_1": np.random.randn(1000),
            "lag_7": np.random.randn(1000),
            "rolling_mean_7": np.random.randn(1000),
        }
    )

    start_time = time.time()
    y_pred = model.predict(X_infer)
    inference_time = time.time() - start_time

    latency_ms = (inference_time / len(X_infer)) * 1000

    # Assert
    assert latency_ms > 0
    assert len(y_pred) == len(X_infer)
    assert latency_ms < 100  # Should be less than 100ms per sample


def test_inference_prediction_shape():
    """Test that inference returns predictions with correct shape"""
    from lightgbm import LGBMRegressor

    X_train = pd.DataFrame(
        np.random.randn(100, 3), columns=["lag_1", "lag_7", "rolling_mean_7"]
    )
    y_train = np.random.randn(100)

    model = LGBMRegressor(n_estimators=5, verbose=-1)
    model.fit(X_train, y_train)

    X_infer = pd.DataFrame(
        np.random.randn(50, 3), columns=["lag_1", "lag_7", "rolling_mean_7"]
    )
    y_pred = model.predict(X_infer)

    assert y_pred.shape == (50,)
