import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def calculate_evaluation_metrics(y_true, y_pred):
    """
    Tính toán các chỉ số MSE, MAE, RMSE, MAPE và R2.
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mape": float(mape),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def check_quality_gate(metrics, mape_threshold, r2_threshold):
    """
    Kiểm tra xem các chỉ số có vượt qua ngưỡng chất lượng (Quality Gate) hay không.
    """
    mape = metrics["mape"]
    r2 = metrics["r2"]

    passed = mape <= mape_threshold and r2 >= r2_threshold
    return bool(passed)
