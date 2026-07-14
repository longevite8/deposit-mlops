import pandas as pd
from scipy.stats import ks_2samp


from config import (
    DRIFT_PVALUE_THRESHOLD,
    DRIFT_RATIO_WARNING_THRESHOLD,
    DRIFT_RATIO_THRESHOLD,
    DRIFT_STATUS_LEVELS,
    FEATURE_COLUMNS,
)


def monitor_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Detect drift between reference and current datasets using the Kolmogorov–Smirnov test.

    Args:
        reference_df (pd.DataFrame): The reference dataset.
        current_df (pd.DataFrame): The current dataset to compare against the reference.

    Returns:
        dict: A dictionary containing the drift detection results.
    """
    drift_rows = []
    n_drift_features = 0

    for col in FEATURE_COLUMNS:
        statistic, p_value = ks_2samp(
            reference_df[col],
            current_df[col],
        )

        is_drift = p_value < DRIFT_PVALUE_THRESHOLD

        if is_drift:
            n_drift_features += 1

        drift_rows.append(
            {
                "feature": col,
                "ks_statistic": statistic,
                "p_value": p_value,
                "drift": is_drift,
            }
        )

    drift_ratio = n_drift_features / len(FEATURE_COLUMNS)
    # Tối ưu logic check bằng cách sử dụng constants và mảng levels
    if drift_ratio < DRIFT_RATIO_WARNING_THRESHOLD:
        status = DRIFT_STATUS_LEVELS[0]  # "PASS"
    elif drift_ratio < DRIFT_RATIO_THRESHOLD:
        status = DRIFT_STATUS_LEVELS[1]  # "WARNING"
    else:
        status = DRIFT_STATUS_LEVELS[2]  # "FAIL"

    drift_result = {
        "status": status,
        "n_features": len(FEATURE_COLUMNS),
        "n_drift_features": n_drift_features,
        "drift_ratio": drift_ratio,
        "drift_data": drift_rows,
    }

    return drift_result
