import time
import joblib
import pandas as pd

from clearml import (
    Task,
    Model,
    Dataset,
)
from pathlib import Path

from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_INFERENCE_NAME,
    FEATURE_COLUMNS,
)

from helpers import wait_for_artifact  # THÊM: Import từ helper

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_INFERENCE_NAME,
    task_type=Task.TaskTypes.inference,
)


# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "feature_task_id": "",
    }
)

# =====================================================
# Template creation mode
# =====================================================

if not params["feature_task_id"]:
    task.get_logger().report_text("Template creation mode.")

    task.close()

    raise SystemExit(0)


# =====================================================
# Load feature dataset
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

feature_dataset = Dataset.get(
    dataset_id=feature_dataset_id,
)

local_path = Path(feature_dataset.get_local_copy())

latest_df = pd.read_parquet(local_path / "test.parquet")

# =====================================================
# Load model
# =====================================================

champion_models = Model.query_models(
    tags=["champion"],
    only_published=True,
    max_results=1,
)

if len(champion_models) == 0:
    raise ValueError("No champion model found.")

champion_model = champion_models[0]

model_path = champion_model.get_local_copy()

model = joblib.load(
    model_path,
)


# =====================================================
# Predict
# =====================================================

X = latest_df[FEATURE_COLUMNS]

start_time = time.time()
prediction = model.predict(X)
inference_time = time.time() - start_time
inference_latency_ms = (inference_time / len(X)) * 1000  # ms per sample


# =====================================================
# Build prediction dataframe
# =====================================================

prediction_df = latest_df.copy()

prediction_df["prediction"] = prediction


# =====================================================
# Upload artifact
# =====================================================

task.upload_artifact(
    "model_id",
    champion_model.id,
)

task.upload_artifact(
    "feature_dataset_id",
    feature_dataset_id,
)

task.upload_artifact(
    name="prediction_df",
    artifact_object=prediction_df,
)

champion_metadata = champion_model.get_all_metadata()
prediction_lineage = {
    "model_id": champion_model.id,
    "feature_dataset_id": feature_dataset_id,
    "raw_dataset_id": champion_metadata.get("raw_dataset_id", {}).get("value", ""),
    "train_task_id": champion_metadata.get("train_task_id", {}).get("value", ""),
    "feature_task_id": champion_metadata.get("feature_task_id", {}).get("value", ""),
    "inference_task_id": task.id,
}

task.upload_artifact(
    "prediction_lineage",
    prediction_lineage,
)

prediction_summary = {
    "n_rows": len(prediction_df),
    "prediction_mean": float(prediction.mean()),
    "prediction_min": float(prediction.min()),
    "prediction_max": float(prediction.max()),
    "prediction_std": float(prediction.std()),
    "inference_time_sec": float(inference_time),
    "latency_ms_per_sample": float(inference_latency_ms),
    "batch_size": len(X),
}

task.upload_artifact(
    "prediction_summary",
    prediction_summary,
)

# =====================================================
# Log
# =====================================================

task.get_logger().report_table(
    title="Prediction",
    series="prediction_df",
    iteration=0,
    table_plot=prediction_df.head(100),  # Hiển thị top 100 rows
)

# Prediction statistics
task.get_logger().report_single_value(
    "prediction_mean",
    float(prediction.mean()),
)

task.get_logger().report_single_value(
    "prediction_min",
    float(prediction.min()),
)

task.get_logger().report_single_value(
    "prediction_max",
    float(prediction.max()),
)

task.get_logger().report_single_value(
    "prediction_std",
    float(prediction.std()),
)

# Inference performance metrics
task.get_logger().report_single_value(
    "total_inference_time_sec",
    float(inference_time),
)

task.get_logger().report_single_value(
    "latency_ms_per_sample",
    float(inference_latency_ms),
)

task.get_logger().report_single_value(
    "inference_batch_size",
    len(X),
)

# Markdown dashboard
task.get_logger().report_text(
    f"""
# Inference Results

## Model & Data
| Field | Value |
|-------|-------|
| Champion Model ID | {champion_model.id} |
| Feature Dataset ID | {feature_dataset_id} |
| Training Task ID | {prediction_lineage["train_task_id"]} |

## Predictions
| Metric | Value |
|--------|-------|
| Mean | {prediction.mean():.4f} |
| Min | {prediction.min():.4f} |
| Max | {prediction.max():.4f} |
| Std Dev | {prediction.std():.4f} |

## Performance
| Metric | Value |
|--------|-------|
| Total Time (sec) | {inference_time:.4f} |
| Latency (ms/sample) | {inference_latency_ms:.4f} |
| Batch Size | {len(X)} |
"""
)

print(prediction_df.tail())

# THÊM: Đồng bộ hoàn toàn trước khi kết thúc
task.flush()

task.close()
