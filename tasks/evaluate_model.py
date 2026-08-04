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

# ✅ Dùng wait_for_artifact để đảm bảo model_id đã được upload hoàn tất
try:
    model_id = wait_for_artifact(
        train_task,
        "model_id",
        max_retries=15,  # Tăng số lần thử
        wait_interval=3.0,
        logger_obj=task,
    )
except ValueError as exc:
    available_artifacts = list(train_task.artifacts.keys())
    task.get_logger().report_text(f"❌ Error getting model_id: {exc!s}")
    task.get_logger().report_text(
        f"📍 Available artifacts in train task: {available_artifacts}"
    )
    raise

input_model = InputModel(model_id=model_id)

model_path = input_model.get_local_copy()

model = joblib.load(model_path)

# =====================================================
# Detect Model Type
# =====================================================

model_type = params.get("model_type", "").strip().lower()

if not model_type:
    try:
        training_summary = wait_for_artifact(
            train_task,
            "training_summary",
            max_retries=10,
            wait_interval=2.0,
            logger_obj=task,
        )
        model_type = str(training_summary["model_type"]).lower()
        task.get_logger().report_text(
            f"✅ Got model_type from training_summary: {model_type}"
        )
    except (KeyError, ValueError) as exc:
        task.get_logger().report_text(
            f"❌ Failed to load model_type from training_summary: {exc!s}"
        )
        task.close(status="failed")
        raise

if model_type not in ["lightgbm", "nhits", "nbeatsx"]:
    task.get_logger().report_text(f"❌ Unsupported model_type: {model_type}")
    task.close(status="failed")
    raise SystemExit(1)

task.get_logger().report_text(f"📊 Model Type: {model_type}")
task.upload_artifact("model_type", model_type)

# =====================================================
# Predict (Model-Specific)
# =====================================================

task.get_logger().report_text(f"🔮 Generating predictions with {model_type.upper()}...")

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
    "feature_dataset_id": feature_dataset_id,
    "passed": passed,
    "mape_threshold": mape_threshold,
    "r2_threshold": r2_threshold,
    "forecast_horizon": forecast_horizon,  # Track horizon in summary
    "num_test_samples": len(y_true),
    **metrics,  # Trộn các metrics (mape, mae, rmse, r2) vào summary
}

evaluate_lineage = {
    "evaluate_task_id": task.id,
    "train_task_id": params["train_task_id"],
    "feature_task_id": params["feature_task_id"],
    "model_id": model_id,
    "feature_dataset_id": feature_dataset_id,
    "model_type": model_type,
    "forecast_horizon": forecast_horizon,  # Track which horizon was used
    "num_test_samples": len(y_true),
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
