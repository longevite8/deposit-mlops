from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clearml import (
    Task,
    Dataset,
)
from pathlib import Path

import pandas as pd


from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_MONITORING_NAME,
    TARGET_COLUMN,
    MONITORING_MAPE_THRESHOLD,
    MONITORING_R2_THRESHOLD,
)

from helpers import wait_for_artifact
from business.monitoring import (
    calculate_monitoring_metrics,
    check_retraining_condition,
)  # THÊM
from business.inference import (
    calculate_prediction_statistics,
)  # Tái sử dụng stats từ bước inference nếu cần

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_MONITORING_NAME,
    task_type=Task.TaskTypes.qc,
)


# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "feature_task_id": "",
        "inference_task_id": "",
        "drift_task_id": "",
    }
)

# =====================================================
# Template creation mode
# =====================================================

if (
    not params["feature_task_id"]
    or not params["inference_task_id"]
    or not params["drift_task_id"]
):
    task.get_logger().report_text("Template creation mode.")

    task.close()

    raise SystemExit(0)


# =====================================================
# Load actual data
# =====================================================

feature_task = Task.get_task(task_id=params["feature_task_id"])

# SỬA: Dùng wait_for_artifact để chắc chắn dataset ID sẵn sàng
feature_dataset_id = wait_for_artifact(
    feature_task,
    "feature_dataset_id",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

feature_dataset = Dataset.get(dataset_id=feature_dataset_id)

local_path = Path(feature_dataset.get_local_copy())

actual_df = pd.read_parquet(local_path / "test.parquet")

# =====================================================
# Load prediction
# =====================================================

inference_task = Task.get_task(task_id=params["inference_task_id"])

# SỬA: Đổi từ prediction_df/summary/lineage -> inference_df/summary/lineage
prediction_df = wait_for_artifact(
    inference_task,
    "prediction_df",  # Giữ nguyên vì đây là data object
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

inference_summary = wait_for_artifact(
    inference_task,
    "inference_summary",  # SỬA phù hợp với task name
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

inference_lineage = wait_for_artifact(
    inference_task,
    "inference_lineage",  # SỬA phù hợp với task name
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

# =====================================================
# Load drift summary
# =====================================================

drift_task = Task.get_task(task_id=params["drift_task_id"])

drift_summary = wait_for_artifact(
    drift_task,
    "drift_summary",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

# SỬA: Lấy drift_result trực tiếp từ drift_summary thay vì đợi một artifact riêng lẻ
drift_result = drift_summary.get("drift_result", {})

drift_lineage = wait_for_artifact(
    drift_task,
    "drift_lineage",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

drift_ratio = drift_summary["drift_ratio"]

drift_status = drift_summary["status"]

# =====================================================
# Ground truth vs Prediction
# =====================================================

y_true = actual_df[TARGET_COLUMN]
y_pred = prediction_df["prediction"]

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

monitoring_metrics_values = calculate_monitoring_metrics(y_true, y_pred)

pred_stats = calculate_prediction_statistics(y_pred)

need_retraining = check_retraining_condition(
    metrics=monitoring_metrics_values,
    drift_status=drift_status,
    mape_threshold=MONITORING_MAPE_THRESHOLD,
    r2_threshold=MONITORING_R2_THRESHOLD,
)

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

# =====================================================
# Monitoring report
# =====================================================

monitoring_summary = {
    "status": ("FAIL" if need_retraining else "PASS"),
    "need_retraining": need_retraining,
    "drift_status": drift_status,
    "drift_ratio": float(drift_ratio),
    "model_id": inference_lineage["model_id"],
    "feature_dataset_id": feature_dataset_id,
}

monitoring_metrics = {**monitoring_metrics_values, **pred_stats}

monitoring_lineage = {
    "model_id": inference_lineage["model_id"],
    "feature_dataset_id": feature_dataset_id,
    "feature_task_id": feature_task.id,
    "inference_task_id": inference_task.id,
    "drift_task_id": drift_task.id,
    "monitoring_task_id": task.id,
}

# =====================================================
# Upload artifact
# =====================================================

task.upload_artifact(
    "monitoring_summary",
    monitoring_summary,
)

task.upload_artifact(
    "monitoring_metrics",
    monitoring_metrics,
)

task.upload_artifact(
    "monitoring_lineage",
    monitoring_lineage,
)

# =====================================================
# Scalars
# =====================================================

task.get_logger().report_single_value(
    "MAPE",
    float(monitoring_metrics["mape"]),
)

task.get_logger().report_single_value(
    "MAE",
    float(monitoring_metrics["mae"]),
)

task.get_logger().report_single_value(
    "RMSE",
    float(monitoring_metrics["rmse"]),
)

task.get_logger().report_single_value(
    "R2",
    float(monitoring_metrics["r2"]),
)

task.get_logger().report_single_value(
    "need_retraining",
    int(need_retraining),
)

task.get_logger().report_single_value(
    "prediction_mean",
    float(monitoring_metrics["prediction_mean"]),
)

task.get_logger().report_single_value(
    "prediction_std",
    float(monitoring_metrics["prediction_std"]),
)

task.get_logger().report_single_value(
    "prediction_min",
    float(monitoring_metrics["prediction_min"]),
)

task.get_logger().report_single_value(
    "prediction_max",
    float(monitoring_metrics["prediction_max"]),
)

task.get_logger().report_single_value(
    "drift_ratio",
    float(drift_ratio),
)


task.get_logger().report_text(
    f"""

    Monitoring Summary

    Status

    {monitoring_summary["status"]}

    Need Retraining

    {monitoring_summary["need_retraining"]}

    Model

    {monitoring_summary["model_id"]}

    Feature Dataset

    {monitoring_summary["feature_dataset_id"]}

    Drift Status

    {monitoring_summary["drift_status"]}

    Drift Ratio

    {monitoring_summary["drift_ratio"]:.3f}

    """
)

markdown = f"""
# Monitoring Dashboard

---

# Monitoring Status

| Item | Value |
|------|------|
| Status | **{monitoring_summary["status"]}** |
| Need Retraining | **{monitoring_summary["need_retraining"]}** |

---

# Model Information

| Item | Value |
|------|------|
| Model ID | {monitoring_summary["model_id"]} |
| Feature Dataset | {monitoring_summary["feature_dataset_id"]} |

---

# Prediction Metrics

| Metric | Value |
|---------|------:|
| MAPE | {monitoring_metrics["mape"]:.4f} |
| MAE | {monitoring_metrics["mae"]:.4f} |
| RMSE | {monitoring_metrics["rmse"]:.4f} |
| R² | {monitoring_metrics["r2"]:.4f} |

---

# Prediction Distribution

| Metric | Value |
|---------|------:|
| Mean | {monitoring_metrics["prediction_mean"]:.4f} |
| Std | {monitoring_metrics["prediction_std"]:.4f} |
| Min | {monitoring_metrics["prediction_min"]:.4f} |
| Max | {monitoring_metrics["prediction_max"]:.4f} |

---

# Drift

| Metric | Value |
|---------|------:|
| Drift Status | {monitoring_summary["drift_status"]} |
| Drift Ratio | {monitoring_summary["drift_ratio"]:.4f} |

"""

task.get_logger().report_text(markdown)

task.get_logger().report_histogram(
    title="Prediction Distribution",
    series="prediction",
    values=y_pred.values,
    iteration=0,
)

summary_df = pd.DataFrame([monitoring_summary])
task.get_logger().report_table(
    title="Monitoring Summary",
    series="summary",
    iteration=0,
    table_plot=summary_df,
)

metrics_df = pd.DataFrame([monitoring_metrics])
task.get_logger().report_table(
    title="Monitoring Metrics",
    series="metrics",
    iteration=0,
    table_plot=metrics_df,
)

prediction_summary_df = pd.DataFrame([inference_summary])
task.get_logger().report_table(
    title="Prediction Summary",
    series="prediction",
    iteration=0,
    table_plot=prediction_summary_df,
)

drift_df = pd.DataFrame.from_dict(
    drift_result,
    orient="index",
).reset_index()

drift_df.rename(
    columns={
        "index": "feature",
    },
    inplace=True,
)

task.get_logger().report_table(
    title="Feature Drift",
    series="drift",
    iteration=0,
    table_plot=drift_df,
)

print(monitoring_summary)


task.flush()
task.close()
