import joblib
import pandas as pd

from clearml import (
    Dataset,
    InputModel,
    Task,
)
from pathlib import Path


from config import (
    FEATURE_COLUMNS,
    MAPE_THRESHOLD,
    PROJECT_TEMPLATE,
    R2_THRESHOLD,
    TARGET_COLUMN,
    TEMPLATE_EVALUATE_NAME,
)

from helpers import wait_for_artifact
from business.evaluate import calculate_evaluation_metrics, check_quality_gate

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
