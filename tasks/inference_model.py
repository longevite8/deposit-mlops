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

from helpers import wait_for_artifact
from business.inference import (
    run_model_inference,
    calculate_prediction_statistics,
    build_output_dataframe,
)  # THÊM

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
    tags=["champion"],
    only_published=True,
    max_results=1,
)

if len(champion_models) == 0:
    raise ValueError("No champion model found.")

champion_model = champion_models[0]
task.get_logger().report_text(f"✅ Loaded champion model: {champion_model.id}")

model_path = champion_model.get_local_copy()

model = joblib.load(
    model_path,
)

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

X = latest_df[FEATURE_COLUMNS]

# Gọi logic inference từ business layer
prediction, inference_time, inference_latency_ms = run_model_inference(model, X)

# Tính toán stats từ business layer
pred_stats = calculate_prediction_statistics(prediction)

# Xây dựng dataframe kết quả từ business layer
prediction_df = build_output_dataframe(latest_df, prediction)

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

task.get_logger().report_text(
    f"✅ Inference completed: {len(X)} samples in {inference_time:.4f}s"
)


# Upload artifact
champion_metadata = champion_model.get_all_metadata()
inference_lineage = {
    "model_id": champion_model.id,
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
    "batch_size": len(X),
}

task.upload_artifact(name="prediction_df", artifact_object=prediction_df)
task.upload_artifact("inference_summary", inference_summary)
task.upload_artifact("inference_lineage", inference_lineage)

# =====================================================
# Log & Markdown Dashboard
# =====================================================

# Log scalars sử dụng dữ liệu từ inference_summary
for key, val in inference_summary.items():
    if key != "batch_size":  # Ví dụ: bỏ qua batch_size nếu đã log riêng
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
| Batch Size | {len(X)} |
"""
)

print(prediction_df.tail())

task.flush()
task.close()
