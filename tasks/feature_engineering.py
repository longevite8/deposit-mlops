import os
import tempfile
import time

from clearml import (
    Task,
    Dataset,
)

from src.features import create_features

from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_FEATURE_NAME,
    FEATURE_COLUMNS,
    TRAIN_RATIO,
    VALID_RATIO,
    PROJECT_DATASET,
)

from helpers import wait_for_artifact  # THÊM: Import từ helper

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_FEATURE_NAME,
    task_type=Task.TaskTypes.data_processing,
)

# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "extract_task_id": "",
    }
)

# =====================================================
# Template creation mode
# =====================================================

if not params["extract_task_id"]:
    task.get_logger().report_text("Template creation mode.")

    task.close()
    raise SystemExit(0)

# =====================================================
# Load raw data
# =====================================================

extract_task = Task.get_task(
    task_id=params["extract_task_id"],
)

# SỬA: Dùng wait_for_artifact để chắc chắn data sẵn sàng
df = wait_for_artifact(
    extract_task,
    "raw_data",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

raw_dataset_id = wait_for_artifact(
    extract_task,
    "raw_dataset_id",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

try:
    # Thử lấy Dataset theo ID từ artifact
    raw_dataset = Dataset.get(dataset_id=raw_dataset_id)
    task.get_logger().report_text(f"✅ Raw dataset loaded: {raw_dataset.id}")
except Exception as e:
    task.get_logger().report_text(
        f"⚠️ Cannot get dataset by ID ({raw_dataset_id}): {str(e)}\n"
        f"   Falling back to latest finalized by name...",
        level="warning",
    )
    # Fallback: Lấy version finalized mới nhất theo tên
    raw_dataset = Dataset.get(
        dataset_project=PROJECT_DATASET,
        dataset_name="Deposit Raw Dataset",  # Đảm bảo tên này khớp với tên đặt trong extract_data.py
    )
    task.get_logger().report_text(f"✅ Raw dataset loaded (fallback): {raw_dataset.id}")

if not raw_dataset:
    raise ValueError(f"Could not find a valid Raw Dataset for ID: {raw_dataset_id}")

# =====================================================
# Feature Engineering
# =====================================================

df = create_features(df)

# =====================================================
# Train / Valid / Test split
# =====================================================

n = len(df)

train_end = int(n * TRAIN_RATIO)

valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

train_df = df.iloc[:train_end]

valid_df = df.iloc[train_end:valid_end]

test_df = df.iloc[valid_end:]

# =====================================================
# Build Feature Dataset
# =====================================================

tmp_dir = tempfile.mkdtemp()

train_file = os.path.join(
    tmp_dir,
    "train.parquet",
)

valid_file = os.path.join(
    tmp_dir,
    "valid.parquet",
)

test_file = os.path.join(
    tmp_dir,
    "test.parquet",
)

train_df.to_parquet(
    train_file,
    index=False,
)

valid_df.to_parquet(
    valid_file,
    index=False,
)

test_df.to_parquet(
    test_file,
    index=False,
)

task.get_logger().report_text(
    f"✅ Created train/valid/test splits:\n"
    f"   Train: {len(train_df)} rows\n"
    f"   Valid: {len(valid_df)} rows\n"
    f"   Test: {len(test_df)} rows"
)

# =====================================================
# Ensure Task is synchronized with Server before creating Dataset
# =====================================================

task.flush()

# =====================================================
# Build Feature Dataset (with retry logic)
# =====================================================

max_retries = 3
feature_dataset = None

for attempt in range(max_retries):
    try:
        task.get_logger().report_text(
            f"📌 Creating feature dataset (attempt {attempt + 1}/{max_retries})..."
        )

        feature_dataset = Dataset.create(
            dataset_project=PROJECT_DATASET,
            dataset_name="Deposit Feature Dataset",
            parent_datasets=[raw_dataset],
        )

        task.get_logger().report_text(
            f"✅ Feature dataset created: {feature_dataset.id}"
        )
        break

    except Exception as e:
        if attempt < max_retries - 1:
            task.get_logger().report_text(
                f"⚠️ Attempt {attempt + 1} failed: {str(e)}\n"
                f"   Retrying in 2 seconds...",
                level="warning",
            )
            time.sleep(2)
        else:
            task.get_logger().report_text(
                f"⚠️ Cannot create new dataset after {max_retries} attempts.\n"
                f"   Getting existing dataset by name...",
                level="warning",
            )
            feature_dataset = Dataset.get(
                dataset_project=PROJECT_DATASET,
                dataset_name="Deposit Feature Dataset",
            )
            task.get_logger().report_text(
                f"✅ Using existing feature dataset: {feature_dataset.id}"
            )

if not feature_dataset:
    raise ValueError("Could not create or get feature dataset")

# =====================================================
# Add files and upload Dataset
# =====================================================

feature_dataset.add_files(tmp_dir)

task.get_logger().report_text("📤 Uploading dataset files...")
feature_dataset.upload()

task.get_logger().report_text("📌 Finalizing dataset...")
feature_dataset.finalize()

task.get_logger().report_text(f"✅ Dataset finalized: {feature_dataset.id}")

# =====================================================
# QUAN TRỌNG: Flush ngay sau khi finalize Dataset
# =====================================================

time.sleep(1)  # Chờ ClearML server xử lý
task.flush()

# =====================================================
# Verify Dataset ID trước khi upload artifact
# =====================================================

try:
    # Verify dataset ID có valid không
    verify_dataset = Dataset.get(dataset_id=feature_dataset.id)
    task.get_logger().report_text(f"✅ Dataset ID verified: {verify_dataset.id}")
    final_dataset_id = verify_dataset.id
except Exception as e:
    task.get_logger().report_text(
        f"⚠️ Dataset ID verification failed: {str(e)}\n"
        f"   Using dataset ID from object anyway: {feature_dataset.id}",
        level="warning",
    )
    final_dataset_id = feature_dataset.id

# =====================================================
# Upload artifacts
# =====================================================

task.upload_artifact(
    "raw_dataset_id",
    raw_dataset_id,
)

task.upload_artifact(
    "feature_dataset_id",
    final_dataset_id,  # ← Dùng verified dataset ID
)

feature_dataset_info = {
    "dataset_id": final_dataset_id,
    "raw_dataset_id": raw_dataset_id,
    "train_rows": len(train_df),
    "valid_rows": len(valid_df),
    "test_rows": len(test_df),
    "feature_columns": FEATURE_COLUMNS,
}

task.upload_artifact(
    "feature_dataset_info",
    feature_dataset_info,
)

# =====================================================
# Logging
# =====================================================

task.get_logger().report_text(f"Features = {FEATURE_COLUMNS}")

task.get_logger().report_text(
    f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}"
)

task.get_logger().report_text(f"Feature Dataset ID = {final_dataset_id}")

task.get_logger().report_text(f"Raw Dataset ID = {raw_dataset_id}")

task.get_logger().report_text(
    f"""
    Feature Dataset
    ------------------------
    id : {final_dataset_id}

    train : {len(train_df)}

    valid : {len(valid_df)}

    test : {len(test_df)}
    """
)

print("✅ Feature engineering completed.")

# =====================================================
# Final flush before closing
# =====================================================

task.flush()

task.close()

# =====================================================
# Upload feature summary and lineage
# =====================================================

feature_summary = {
    "feature_dataset_id": feature_dataset.id,
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "features": FEATURE_COLUMNS,
}
feature_lineage = {
    "feature_task_id": task.id,
    "extract_task_id": params["extract_task_id"],
    "feature_dataset_id": feature_dataset.id,
}
task.upload_artifact("feature_summary", feature_summary)
task.upload_artifact("feature_lineage", feature_lineage)
