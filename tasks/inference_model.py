from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numbers import Number
import time
import pandas as pd

from clearml import (
    Task,
    Model,
    Dataset,
)
from pathlib import Path

from config import (
    FORECAST_HORIZON,
    FORECAST_UNIQUE_ID,
    PROJECT_TEMPLATE,
    TEMPLATE_INFERENCE_NAME,
    forecast_horizon_tag,
)

from helpers import wait_for_artifact
from business.inference import calculate_prediction_statistics
from business.forecasting import load_forecast_model_from_archive, prepare_forecast_frame

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
        "horizon": FORECAST_HORIZON,
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
# Load feature dataset - Với error handling tốt hơn
# =====================================================

feature_task = Task.get_task(task_id=params["feature_task_id"])

# Dùng wait_for_artifact để chắc chắn artifact sẵn sàng
feature_dataset_id = wait_for_artifact(
    feature_task,
    "feature_dataset_id",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

task.get_logger().report_text(f"✅ Feature Dataset ID: {feature_dataset_id}")

# Thêm error handling để debug
try:
    feature_dataset = Dataset.get(
        dataset_id=feature_dataset_id,
    )
    task.get_logger().report_text(
        f"✅ Dataset loaded successfully: {feature_dataset.id}"
    )
except ValueError as e:
    task.get_logger().report_text(
        f"❌ Error loading dataset: {str(e)}\n"
        f"   Feature Dataset ID: {feature_dataset_id}\n"
        f"   Attempting to get dataset from feature task artifacts...",
        level="error",
    )

    # Thử lấy dataset từ feature task artifacts
    try:
        # Lấy tất cả artifacts của feature task
        feature_task_artifacts = feature_task.artifacts

        task.get_logger().report_text(
            f"📋 Available artifacts from feature task:\n"
            f"{', '.join([a for a in feature_task_artifacts.keys()])}"
        )

        # Thử lấy dataset ID từ artifact với tên khác
        if "feature_dataset_id" in feature_task_artifacts:
            feature_dataset_id = feature_task_artifacts["feature_dataset_id"].get()
            task.get_logger().report_text(
                f"🔍 Retrieved feature_dataset_id: {feature_dataset_id}"
            )

        # Retry lấy dataset
        feature_dataset = Dataset.get(
            dataset_id=feature_dataset_id,
        )
        task.get_logger().report_text(
            f"✅ Dataset loaded after retry: {feature_dataset.id}"
        )
    except Exception as e2:
        task.get_logger().report_text(
            f"❌ Failed to load dataset even after retry: {str(e2)}", level="error"
        )
        raise

local_path = Path(feature_dataset.get_local_copy())

# Check xem file parquet có tồn tại không
parquet_file = local_path / "test.parquet"
if not parquet_file.exists():
    task.get_logger().report_text(
        f"⚠️ File not found: {parquet_file}\n"
        f"Available files: {list(local_path.glob('*'))}",
        level="warning",
    )
    # Thử tìm parquet file khác
    parquet_files = list(local_path.glob("*.parquet"))
    if parquet_files:
        parquet_file = parquet_files[0]
        task.get_logger().report_text(f"📌 Using parquet file: {parquet_file}")

latest_df = pd.read_parquet(parquet_file)
task.get_logger().report_text(f"✅ Loaded {len(latest_df)} rows from feature dataset")

# =====================================================
# Load model
# =====================================================

champion_models = Model.query_models(
    tags=["champion", forecast_horizon_tag(int(params["horizon"]))],
    only_published=True,
    max_results=1,
)

if len(champion_models) == 0:
    raise ValueError(f"No champion model found for horizon {params['horizon']}.")

champion_model = champion_models[0]
task.get_logger().report_text(f"✅ Loaded champion model: {champion_model.id}")

model_path = champion_model.get_local_copy()
if not model_path:
    raise ValueError(f"Could not resolve local model archive for model_id={champion_model.id}")

model = load_forecast_model_from_archive(model_path)

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

history_df = prepare_forecast_frame(latest_df, unique_id=FORECAST_UNIQUE_ID)

start_time = time.time()
forecasts = model.predict(df=history_df)
inference_time = time.time() - start_time

forecast_columns = [
    column
    for column in forecasts.columns
    if column not in {"unique_id", "ds", "cutoff", "y", "step", "month", "weekday"}
    and pd.api.types.is_numeric_dtype(forecasts[column])
]
if not forecast_columns:
    raise ValueError(f"No numeric forecast columns found: {list(forecasts.columns)}")

champion_metadata = champion_model.get_all_metadata()
metadata_primary_model = champion_metadata.get("primary_model", {}).get("value")
primary_model = (
    metadata_primary_model
    if metadata_primary_model in forecast_columns
    else forecast_columns[0]
)
prediction = forecasts[primary_model].to_numpy(dtype=float)
inference_latency_ms = (
    (inference_time / len(prediction)) * 1000 if len(prediction) > 0 else 0
)
pred_stats = calculate_prediction_statistics(prediction)
prediction_df = forecasts.copy()
prediction_df["prediction"] = prediction

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

task.get_logger().report_text(
    f"✅ Inference completed: {len(prediction)} forecasts in {inference_time:.4f}s"
)


# Upload artifact
inference_lineage = {
    "model_id": champion_model.id,
    "forecast_horizon": int(params["horizon"]),
    "feature_dataset_id": feature_dataset_id,
    "raw_dataset_id": champion_metadata.get("raw_dataset_id", {}).get("value", ""),
    "train_task_id": champion_metadata.get("train_task_id", {}).get("value", ""),
    "feature_task_id": champion_metadata.get("feature_task_id", {}).get("value", ""),
    "inference_task_id": task.id,
}

# Tạo inference_summary bằng cách trộn kết hợp stats và performance
inference_summary = {
    **pred_stats,
    "total_inference_time_sec": float(inference_time),
    "latency_ms_per_sample": float(inference_latency_ms),
    "batch_size": len(prediction),
    "primary_model": primary_model,
}

task.upload_artifact(name="prediction_df", artifact_object=prediction_df)
task.upload_artifact("inference_summary", inference_summary)
task.upload_artifact("inference_lineage", inference_lineage)

# =====================================================
# Log & Markdown Dashboard
# =====================================================

# Log scalars sử dụng dữ liệu từ inference_summary
for key, val in inference_summary.items():
    if key != "batch_size" and isinstance(val, Number):
        task.get_logger().report_single_value(key, val)

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
    len(prediction),
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
| Training Task ID | {inference_lineage["train_task_id"]} |

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
| Batch Size | {len(prediction)} |
"""
)

print(prediction_df.tail())

task.flush()
task.close()
