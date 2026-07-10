import joblib
import numpy as np
import pandas as pd

from clearml import (
    Dataset,
    InputModel,
    Task,
)
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from config import (
    FEATURE_COLUMNS,
    MAPE_THRESHOLD,
    PROJECT_TEMPLATE,
    R2_THRESHOLD,
    TARGET_COLUMN,
    TEMPLATE_EVALUATE_NAME,
)

from helpers import wait_for_artifact  # THÊM: Import từ helper

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

# SỬA: Dùng wait_for_artifact để chắc chắn dataset ID sẵn sàng
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


train_task = Task.get_task(task_id=params["train_task_id"])

model_id = train_task.artifacts["model_id"].get()


input_model = InputModel(model_id=model_id)

model_path = input_model.get_local_copy()

model = joblib.load(model_path)


# =====================================================
# Predict
# =====================================================

y_pred = model.predict(X_test)


# =====================================================
# Metrics
# =====================================================

mape = mean_absolute_percentage_error(
    y_true,
    y_pred,
)

mae = mean_absolute_error(
    y_true,
    y_pred,
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred,
    )
)

r2 = r2_score(
    y_true,
    y_pred,
)


# =====================================================
# Quality gate
# =====================================================

passed = mape <= mape_threshold and r2 >= r2_threshold


# =====================================================
# Upload evaluation result
# =====================================================

raw_dataset_id = train_task.artifacts["raw_dataset_id"].get()

task.upload_artifact(
    "raw_dataset_id",
    raw_dataset_id,
)

task.upload_artifact(
    "feature_dataset_id",
    feature_dataset_id,
)

evaluation_summary = {
    "Feature Dataset": feature_dataset_id,
    "Raw Dataset": raw_dataset_id,
    "mape": float(mape),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "passed": bool(passed),
    "mape_threshold": mape_threshold,
    "r2_threshold": r2_threshold,
}

task.upload_artifact(
    "evaluate_summary",
    evaluation_summary,
)

evaluate_lineage = {
    "evaluate_task_id": task.id,
    "train_task_id": train_task.id,
    "feature_task_id": feature_task.id,
    "feature_dataset_id": feature_dataset_id,
    "raw_dataset_id": raw_dataset_id,
    "model_id": model_id,
}

task.upload_artifact(
    "evaluate_lineage",
    evaluate_lineage,
)

# =====================================================
# Scalars
# =====================================================

task.get_logger().report_single_value("mape", float(mape))

task.get_logger().report_single_value("mae", float(mae))

task.get_logger().report_single_value("rmse", float(rmse))

task.get_logger().report_single_value("r2", float(r2))

task.get_logger().report_single_value("mape_threshold", mape_threshold)

task.get_logger().report_single_value("r2_threshold", r2_threshold)

task.get_logger().report_single_value("quality_gate_passed", int(passed))

task.get_logger().report_text(f"Evaluation result = {evaluation_summary}")

print(evaluation_summary)

# Đồng bộ hoàn toàn trước khi kết thúc
task.flush()

task.close()
