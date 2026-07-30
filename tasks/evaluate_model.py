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

# Import neural models for type checking
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, NBEATSx
except ImportError:
    NHITS = None
    NBEATSx = None
    NeuralForecast = None

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


feature_task = Task.get_task(
    task_id=params["feature_task_id"],
)

# Dùng wait_for_artifact để chắc chắn dataset ID sẵn sàng
feature_dataset_id = wait_for_artifact(
    feature_task,
    "feature_dataset_id",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

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
    # Use target_h (last column = forecast_horizon step) for evaluation
    y_true = test_df[target_cols[-1]]  # Last target column
    task.get_logger().report_text(
        f"✅ Using multi-target ground truth: {target_cols[-1]} (for forecast horizon)"
    )
else:
    # Fallback to single target if multi-targets not available
    task.get_logger().report_text(
        "⚠️ No multi-target columns found, using single target"
    )


train_task = Task.get_task(task_id=params["train_task_id"])
task.get_logger().report_text(
    f"📍 Evaluating model from Train Task ID: {params['train_task_id']}"
)

# ✅ Dùng wait_for_artifact để đảm bảo model_id đã được upload hoàn tất
try:
    model_id = wait_for_artifact(
        train_task,
        "model_id",
        max_retries=15,  # Tăng số lần thử
        wait_interval=3.0,
        logger_obj=task,
    )
except Exception as e:
    # Log chi tiết các artifacts hiện có để debug
    available_artifacts = list(train_task.artifacts.keys())
    task.get_logger().report_text(f"❌ Error getting model_id: {e!s}")
    task.get_logger().report_text(
        f"📍 Available artifacts in train task: {available_artifacts}"
    )
    raise e

input_model = InputModel(model_id=model_id)

model_path = input_model.get_local_copy()

model = joblib.load(model_path)

# =====================================================
# Detect Model Type
# =====================================================

# Try to get model_type from params, train_task, or detect from instance
model_type = params.get("model_type", "").lower()

if not model_type:
    # Try to get from train_task artifact
    try:
        model_type = wait_for_artifact(
            train_task, "model_type", max_retries=5, wait_interval=1.0, logger_obj=task
        )
        model_type = model_type.lower()
    except Exception:
        # Detect from model instance
        model_class_name = model.__class__.__name__.lower()
        if "nhits" in model_class_name:
            model_type = "nhits"
        elif "nbeatsx" in model_class_name:
            model_type = "nbeatsx"
        else:
            model_type = "lightgbm"

task.get_logger().report_text(f"📊 Model Type Detected: {model_type}")

# =====================================================
# Predict (Model-Specific)
# =====================================================

if model_type in ["nhits", "nbeatsx"]:
    # Neural models: need NeuralForecast format (ds, y, unique_id)
    if NeuralForecast is None:
        raise ImportError(
            "NeuralForecast not installed. Cannot predict with neural models."
        )

    task.get_logger().report_text(
        f"📊 Using NeuralForecast prediction for {model_type.upper()}"
    )

    # Prepare test data in NeuralForecast format
    # Create dummy dates and y values
    test_dates = pd.date_range(start="2020-01-01", periods=len(X_test))
    pred_df = pd.DataFrame(
        {
            "ds": test_dates,
            "y": np.zeros(len(X_test)),  # Dummy y values (not used for prediction)
            "unique_id": "target",
        }
    )

    # Create NeuralForecast instance and predict
    nf = NeuralForecast(models=[model], freq="D")
    forecasts = nf.predict(pred_df)

    # Extract predictions
    model_col = model_type.upper()
    if model_col in forecasts.columns:
        y_pred = forecasts[model_col].values
    else:
        # Fallback: get first prediction column
        pred_cols = [col for col in forecasts.columns if col not in ["ds", "unique_id"]]
        if pred_cols:
            y_pred = forecasts[pred_cols[0]].values
        else:
            raise ValueError(
                f"Cannot find prediction column in forecasts. Columns: {forecasts.columns.tolist()}"
            )

else:
    # Tree-based models (LightGBM, etc): standard sklearn predict
    task.get_logger().report_text("📊 Using sklearn-style prediction for LightGBM")
    y_pred = model.predict(X_test)

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
    "feature_dataset_id": feature_dataset_id,
    "passed": passed,
    "mape_threshold": mape_threshold,
    "r2_threshold": r2_threshold,
    **metrics,  # Trộn các metrics (mape, mae, rmse, r2) vào summary
}

evaluate_lineage = {
    "evaluate_task_id": task.id,
    "train_task_id": params["train_task_id"],
    "feature_task_id": params["feature_task_id"],
    "model_id": model_id,
    "feature_dataset_id": feature_dataset_id,
    "model_type": model_type,
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
