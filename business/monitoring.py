import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def calculate_monitoring_metrics(y_true, y_pred):
    """
    Tính toán các chỉ số Performance Monitoring giữa thực tế và dự báo.
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


def check_retraining_condition(metrics, drift_status, mape_threshold, r2_threshold):
    """
    Kiểm tra các điều kiện để quyết định có cần Retrain mô hình hay không.
    """
    mape = metrics["mape"]
    r2 = metrics["r2"]

    # Điều kiện: MAPE quá cao HOẶC R2 quá thấp HOẶC Drift ở mức FAIL
    need_retraining = (
        mape > mape_threshold or r2 < r2_threshold or drift_status == "FAIL"
    )

    return bool(need_retraining)
