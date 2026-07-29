"""
Shared Utilities cho Model Implementations.
Bao gồm data preprocessing, normalization, temporal tensor creation, etc.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_temporal_tensors(
    X: np.ndarray, y: np.ndarray, input_size: int = 8, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create temporal tensors từ flat time series data.
    Reshape từ (n_samples, n_features) to (n_samples, seq_len, n_features).

    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        input_size: Lookback window size
        horizon: Forecast horizon (steps ahead to predict)

    Returns:
        Tuple of (X_temporal, y_temporal):
        - X_temporal: (n_sequences, input_size, n_features)
        - y_temporal: (n_sequences, horizon)
    """
    n_samples = len(X)
    n_features = X.shape[1] if len(X.shape) > 1 else 1

    # Ensure X is 2D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    X_temporal = []
    y_temporal = []

    # Create sequences
    for i in range(n_samples - input_size - horizon + 1):
        X_seq = X[i : i + input_size]  # (input_size, n_features)
        y_seq = y[i + input_size : i + input_size + horizon]  # (horizon,)

        X_temporal.append(X_seq)
        y_temporal.append(y_seq)

    X_temporal = np.array(X_temporal)  # (n_sequences, input_size, n_features)
    y_temporal = np.array(y_temporal)  # (n_sequences, horizon)

    # Handle case where horizon=1
    if y_temporal.shape[-1] == 1:
        y_temporal = y_temporal.squeeze(-1)  # (n_sequences,)

    return X_temporal, y_temporal


def normalize_data(
    X: pd.DataFrame, scaler: StandardScaler | None = None
) -> tuple[np.ndarray, StandardScaler]:
    """
    Normalize data using StandardScaler (zero mean, unit variance).

    Args:
        X: Input data (DataFrame hoặc ndarray)
        scaler: Existing scaler để apply (nếu None, create mới)

    Returns:
        Tuple of (X_normalized, scaler)
    """
    # Convert to ndarray if DataFrame
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X

    # Create hoặc fit scaler
    if scaler is None:
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_values)
    else:
        X_normalized = scaler.transform(X_values)

    return X_normalized, scaler


def inverse_normalize(X_normalized: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Inverse normalize (denormalize) dữ liệu.

    Args:
        X_normalized: Normalized data
        scaler: StandardScaler object với fit() từ original data

    Returns:
        np.ndarray: Denormalized data
    """
    X_original = scaler.inverse_transform(X_normalized)
    return X_original


def create_prediction_features(
    historical_data: pd.DataFrame, feature_columns: list, lookback: int = 8
) -> np.ndarray:
    """
    Create feature matrix từ historical data cho prediction.

    Args:
        historical_data: DataFrame chứa historical data
        feature_columns: List of feature column names
        lookback: Number of time steps to include

    Returns:
        np.ndarray: Feature matrix (lookback, n_features)
    """
    X = historical_data[feature_columns].tail(lookback).values
    return X


def handle_missing_values(
    X: pd.DataFrame, method: str = "forward_fill"
) -> pd.DataFrame:
    """
    Handle missing values trong time series data.

    Args:
        X: Input DataFrame
        method: "forward_fill", "backward_fill", hoặc "interpolate"

    Returns:
        pd.DataFrame: Cleaned data
    """
    X = X.copy()

    if method == "forward_fill":
        X = X.fillna(method="ffill").fillna(method="bfill")
    elif method == "backward_fill":
        X = X.fillna(method="bfill").fillna(method="ffill")
    elif method == "interpolate":
        X = X.interpolate(method="linear", limit_direction="both")
    else:
        raise ValueError(f"Unknown method: {method}")

    return X


def split_temporal_data(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.6, valid_ratio: float = 0.2
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """
    Split temporal data vào train/valid/test while maintaining temporal order.

    Args:
        X, y: Input data
        train_ratio, valid_ratio: Ratio cho train/valid (test = 1 - train - valid)

    Returns:
        Tuple of ((X_train, y_train), (X_valid, y_valid), (X_test, y_test))
    """
    n = len(X)
    train_size = int(n * train_ratio)
    valid_size = int(n * valid_ratio)

    X_train, y_train = X[:train_size], y[:train_size]
    X_valid, y_valid = (
        X[train_size : train_size + valid_size],
        y[train_size : train_size + valid_size],
    )
    X_test, y_test = X[train_size + valid_size :], y[train_size + valid_size :]

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def rolling_forecast_neural(
    nf,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    input_size: int = 8,
) -> np.ndarray:
    """
    Generate rolling forecasts for validation period using NeuralForecast model.

    Since NeuralForecast.predict() only forecasts 1 step ahead (h=1),
    we need to implement rolling forecast to get predictions for entire validation period.

    Args:
        nf: Fitted NeuralForecast instance
        train_df: Training dataframe (ds, y, unique_id)
        valid_df: Validation dataframe (ds, y, unique_id)
        input_size: Input window size (for context)

    Returns:
        np.ndarray: Array of predictions for validation period
    """
    forecasts = []

    # Start with training data as history
    history = train_df[["ds", "y", "unique_id"]].copy()

    # For each validation point
    for idx in range(len(valid_df)):
        try:
            # Predict next step
            pred = nf.predict()

            # Extract prediction value
            pred_value = pred.iloc[0, 0] if isinstance(pred, pd.DataFrame) else pred[0]
            forecasts.append(float(pred_value))

            # Append prediction to history for next iteration
            next_date = history["ds"].max() + pd.Timedelta(days=1)
            new_row = pd.DataFrame(
                {"ds": [next_date], "y": [float(pred_value)], "unique_id": ["target"]}
            )
            history = pd.concat([history, new_row], ignore_index=True)

            # Recreate NeuralForecast with updated history
            nf_updated = NeuralForecast(models=nf.models, freq="D")
            # Note: We fit to update internal state, but this is expensive
            # Alternative: use in-sample predictions from trained model

        except Exception as e:
            print(f"Rolling forecast failed at step {idx}: {e}")
            # Fallback: return what we have so far, padded with NaN
            forecasts.extend([np.nan] * (len(valid_df) - idx))
            break

    return np.array(forecasts)


def use_validation_loss(nf_model) -> float:
    """
    Extract validation loss from trained NeuralForecast model.

    This is faster than rolling forecast and uses the model's internal
    validation loss computed during training.

    Args:
        nf_model: Fitted NeuralForecast model instance

    Returns:
        float: Validation loss value (MAE or other configured loss)
    """
    try:
        # Access PyTorch Lightning trainer
        trainer = nf_model.trainer

        # Try to get validation loss from logged metrics
        if hasattr(trainer, "logged_metrics"):
            val_loss = trainer.logged_metrics.get("val_loss")
            if val_loss is not None:
                return float(val_loss)

        # Alternative: callback metrics
        if hasattr(trainer, "callback_metrics"):
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                return float(val_loss)

        # If neither works, return high penalty
        print("Warning: Could not extract validation loss from trainer")
        return float("inf")

    except Exception as e:
        print(f"Error extracting validation loss: {e}")
        return float("inf")
