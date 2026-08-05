from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from clearml import (
    Dataset,
    InputModel,
    Task,
)

from business.evaluate import calculate_evaluation_metrics, check_quality_gate
from config import (
    FEATURE_COLUMNS,
    MAPE_THRESHOLD,
    PROJECT_TEMPLATE,
    R2_THRESHOLD,
    TARGET_COLUMN,
    TEMPLATE_EVALUATE_NAME,
)
from helpers import wait_for_artifact

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_EVALUATE_NAME,
    task_type=Task.TaskTypes.qc,
)


params = task.connect(
    {
        "feature_task_id": "",
        "train_task_id": "",
        "mape_threshold": MAPE_THRESHOLD,
        "r2_threshold": R2_THRESHOLD,
        "model_type": "",  # Optional: specify model type (lightgbm, nhits, nbeatsx)
    }
)

# =====================================================
# Template creation mode
# =====================================================

if not params["feature_task_id"] or not params["train_task_id"]:
    task.get_logger().report_text("Template creation mode.")

    task.close()
    raise SystemExit(0)

# Allow per-run threshold overrides from the ClearML UI or pipeline parameter_override
mape_threshold: float = float(params["mape_threshold"])
r2_threshold: float = float(params["r2_threshold"])


train_task = Task.get_task(
    task_id=params["train_task_id"],
)

task.get_logger().report_text(
    f"📍 Evaluating model from Train Task ID: {params['train_task_id']}"
)

training_summary = wait_for_artifact(
    train_task,
    "training_summary",
    max_retries=15,
    wait_interval=3.0,
    logger_obj=task,
)

training_lineage = wait_for_artifact(
    train_task,
    "training_lineage",
    max_retries=15,
    wait_interval=3.0,
    logger_obj=task,
)

model_id = training_summary["model_id"]
model_type = str(training_summary["model_type"]).strip().lower()
feature_dataset_id = training_lineage["feature_dataset_id"]

feature_dataset = Dataset.get(
    dataset_id=feature_dataset_id,
)

local_path = Path(feature_dataset.get_local_copy())

test_df = pd.read_parquet(local_path / "test.parquet")


X_test = test_df[FEATURE_COLUMNS]

y_true = test_df[TARGET_COLUMN]

# =====================================================
# Multi-target ground truth extraction (for multi-target strategy)
# =====================================================

# Check if multi-step targets exist (created by create_multistep_targets)
target_cols = [col for col in test_df.columns if col.startswith("target_")]
if target_cols:
    # Multi-target strategy: Use target_h (last column = forecast_horizon step)
    forecast_horizon = len(target_cols)
    y_true = test_df[target_cols[-1]]  # Last target column (target_h)
    task.get_logger().report_text(
        f"✅ Multi-target strategy detected (FORECAST_HORIZON={forecast_horizon})"
    )
    task.get_logger().report_text(f"   Targets: {', '.join(target_cols)}")
    task.get_logger().report_text(
        f"   Using target_h for evaluation: {target_cols[-1]} (step {forecast_horizon})"
    )
    task.get_logger().report_text(f"   Ground truth shape: {y_true.shape}")
else:
    # Fallback to single target if multi-targets not available
    forecast_horizon = 1
    task.get_logger().report_text(
        "⚠️ No multi-target columns found, using single target (FORECAST_HORIZON=1)"
    )
    task.get_logger().report_text(f"   Ground truth shape: {y_true.shape}")


train_task = Task.get_task(task_id=params["train_task_id"])
task.get_logger().report_text(
    f"📍 Evaluating model from Train Task ID: {params['train_task_id']}"
)

input_model = InputModel(model_id=model_id)

model_path = input_model.get_local_copy()

model_artifact = joblib.load(model_path)

if not isinstance(model_artifact, dict):
    raise TypeError(
        "Expected the trained model artifact to be a dict bundle, "
        f"but received {type(model_artifact).__name__}."
    )

artifact_model_type = str(model_artifact.get("model_type", "")).strip().lower()

if artifact_model_type and artifact_model_type != model_type:
    raise ValueError(
        f"Model type mismatch: training_summary says "
        f"'{model_type}', but model artifact says "
        f"'{artifact_model_type}'."
    )

forecast_horizon = int(
    model_artifact.get(
        "forecast_horizon",
        forecast_horizon,
    )
)

task.get_logger().report_text(
    f"📦 Loaded model artifact: type={artifact_model_type}, horizon={forecast_horizon}"
)


if model_type not in ["lightgbm", "nhits", "nbeatsx"]:
    task.get_logger().report_text(f"❌ Unsupported model_type: {model_type}")
    task.close(status="failed")
    raise SystemExit(1)

task.get_logger().report_text(f"📊 Model Type: {model_type}")

# =====================================================
# Predict (Model-Specific)
# =====================================================

task.get_logger().report_text(f"🔮 Generating predictions with {model_type.upper()}...")

if model_type in ["nhits", "nbeatsx"]:
    task.get_logger().report_text(
        f"📊 Using fitted NeuralForecast for {model_type.upper()}"
    )

    nf = model_artifact.get("neural_forecast")
    scaler_y = model_artifact.get("scaler_y")

    if nf is None:
        raise ValueError(
            "The neural model artifact does not contain 'neural_forecast'."
        )

    if scaler_y is None:
        raise ValueError("The neural model artifact does not contain 'scaler_y'.")

    forecasts = nf.predict()

    model_column = "NBEATSx" if model_type == "nbeatsx" else "NHITS"

    if model_column not in forecasts.columns:
        prediction_columns = [
            column for column in forecasts.columns if column not in ["unique_id", "ds"]
        ]

        if not prediction_columns:
            raise ValueError(
                "Cannot find neural prediction column. "
                f"Available columns: {forecasts.columns.tolist()}"
            )

        model_column = prediction_columns[0]

    from business.models.utils import inverse_normalize

    y_pred = inverse_normalize(
        forecasts[model_column].to_numpy().reshape(-1, 1),
        scaler_y,
    ).ravel()

    y_pred = np.asarray(y_pred).reshape(-1)

    task.get_logger().report_text(f"📊 Neural prediction shape: {y_pred.shape}")

else:
    # Tree-based models use the nested model from the artifact bundle.
    task.get_logger().report_text("📊 Using sklearn-style prediction for LightGBM")

    model = model_artifact.get("model")

    if model is None:
        raise ValueError(
            "The LightGBM model artifact does not contain the 'model' key."
        )

    if not hasattr(model, "predict"):
        raise TypeError(
            "The nested LightGBM model does not provide predict(). "
            f"Received {type(model).__name__}."
        )

    y_pred = model.predict(X_test)

    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        task.get_logger().report_text(
            f"📊 Multi-output prediction detected: {y_pred.shape}"
        )
        task.get_logger().report_text(
            f"📊 Extracting target_h from the last output column "
            f"of {y_pred.shape[1]} targets."
        )
        y_pred = y_pred[:, -1]
    else:
        y_pred = np.asarray(y_pred).reshape(-1)

    task.get_logger().report_text(f"📊 LightGBM prediction shape: {y_pred.shape}")

    # Handle multi-output predictions (extract target_h)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        task.get_logger().report_text(
            f"📊 Multi-output prediction detected (shape: {y_pred.shape})"
        )
        task.get_logger().report_text(f"   Model trained on {y_pred.shape[1]} targets")
        task.get_logger().report_text(
            f"   Extracting target_h (column {y_pred.shape[1]})..."
        )
        y_pred = y_pred[:, -1]  # Extract last column (target_h)
        task.get_logger().report_text(
            f"   Prediction shape after extraction: {y_pred.shape}"
        )
    else:
        task.get_logger().report_text(
            f"📊 Single-output prediction (shape: {y_pred.shape})"
        )

# =====================================================
# Validate shapes before metrics calculation
# =====================================================

task.get_logger().report_text("🔍 Validating prediction shapes...")
task.get_logger().report_text(
    f"   y_true shape: {y_true.shape} | y_pred shape: {y_pred.shape}"
)

if len(y_true) != len(y_pred):
    raise ValueError(
        f"❌ Shape mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)}). "
        f"Cannot calculate metrics."
    )

if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
    raise ValueError(
        f"❌ y_pred has unexpected multi-dimensional shape: {y_pred.shape}. "
        f"Expected 1D array after target_h extraction."
    )

task.get_logger().report_text("✅ Shape validation passed")

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

# Gọi logic tính toán từ business layer
metrics = calculate_evaluation_metrics(y_true, y_pred)

# Kiểm tra Quality Gate từ business layer
passed = check_quality_gate(metrics, mape_threshold, r2_threshold)

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

# =====================================================
# Upload evaluation result
# =====================================================

evaluate_summary = {
    "passed": passed,
    "mape_threshold": mape_threshold,
    "r2_threshold": r2_threshold,
    "forecast_horizon": forecast_horizon,
    "model_type": model_type,
    "num_test_samples": len(y_true),
    **metrics,
}

evaluate_lineage = {
    "evaluate_task_id": task.id,
    "train_task_id": params["train_task_id"],
    "feature_task_id": params["feature_task_id"],
    "model_id": model_id,
    "feature_dataset_id": feature_dataset_id,
}

task.upload_artifact("evaluate_summary", evaluate_summary)
task.upload_artifact("evaluate_lineage", evaluate_lineage)

# =====================================================
# Scalars
# =====================================================

# Log các chỉ số lên bảng điều khiển ClearML
for metric_name, metric_value in metrics.items():
    task.get_logger().report_single_value(metric_name, metric_value)

task.get_logger().report_single_value("quality_gate_passed", int(passed))
task.get_logger().report_text(f"Evaluation completed. Passed: {passed}")


task.flush()
task.close()
