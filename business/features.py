"""Feature engineering functions for cashflow forecasting."""

import pandas as pd

from config import (
    DATE_COLUMN,
    FORECAST_HORIZON,
    LAG_FEATURES,
    ROLLING_FEATURES,
    TARGET_COLUMN,
)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag and rolling features from raw cashflow data.

    Input:
      df: DataFrame with DATE_COLUMN and TARGET_COLUMN

    Output:
      df: DataFrame with additional lag/rolling features
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure date column is datetime
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # =====================================================
    # Create lag features
    # =====================================================

    # LAG_FEATURES = [1, 7] — list of lag values
    for lag in LAG_FEATURES:
        df[f"lag_{lag}"] = df[TARGET_COLUMN].shift(lag)

    # =====================================================
    # Create rolling features
    # =====================================================

    # ROLLING_FEATURES = [("rolling_mean", 7), ("rolling_std", 7), ...]
    for feature_name, window in ROLLING_FEATURES:
        if feature_name == "rolling_mean":
            df[f"{feature_name}_{window}"] = (
                df[TARGET_COLUMN].rolling(window=window).mean()
            )
        elif feature_name == "rolling_std":
            df[f"{feature_name}_{window}"] = (
                df[TARGET_COLUMN].rolling(window=window).std()
            )
        # Add more rolling aggregates as needed

    # =====================================================
    # Drop rows with NaN (from lag/rolling operations)
    # =====================================================

    df = df.dropna().reset_index(drop=True)

    # =====================================================
    # Create multi-step targets (for multi-target strategy)
    # =====================================================

    df = create_multistep_targets(df, forecast_horizon=FORECAST_HORIZON)

    return df


def create_multistep_targets(
    df: pd.DataFrame, forecast_horizon: int = 1
) -> pd.DataFrame:
    """
    Create multi-step target columns for tree-based models (LightGBM).

    Multi-target strategy:
    - For forecast_horizon=7, create y_t+1, y_t+2, ..., y_t+7
    - Each column represents "number of days ahead" target
    - Enables LightGBM to predict all horizons in one model

    Args:
        df: DataFrame with TARGET_COLUMN
        forecast_horizon: Number of steps ahead to create targets for

    Returns:
        df: DataFrame with additional target_1, target_2, ..., target_h columns
    """
    df = df.copy()

    # Create shifted targets for each step
    for step in range(1, forecast_horizon + 1):
        df[f"target_{step}"] = df[TARGET_COLUMN].shift(-step)

    # Drop rows where any target is NaN (rows at the end will have NaN)
    target_cols = [f"target_{i}" for i in range(1, forecast_horizon + 1)]
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    return df
