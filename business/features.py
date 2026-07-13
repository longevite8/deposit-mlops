"""Feature engineering functions for cashflow forecasting."""

import pandas as pd
from config import (
    LAG_FEATURES,
    ROLLING_FEATURES,
    TARGET_COLUMN,
    DATE_COLUMN,
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

    return df
