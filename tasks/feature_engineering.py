import os
import tempfile

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

df = extract_task.artifacts["raw_data"].get()

raw_dataset_id = extract_task.artifacts["raw_dataset_id"].get()

raw_dataset = Dataset.get(
    dataset_id=raw_dataset_id,
)

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

# THÊM: Đảm bảo Task đã đồng bộ với Server trước khi tạo Dataset
task.flush()

# SỬA: Logic tạo Dataset an toàn hơn
try:
    feature_dataset = Dataset.create(
        dataset_project=PROJECT_DATASET,
        dataset_name="Deposit Feature Dataset",
        parent_datasets=[raw_dataset],
    )
except Exception:
    feature_dataset = Dataset.get(
        dataset_project=PROJECT_DATASET,
        dataset_name="Deposit Feature Dataset",
    )

feature_dataset.add_files(
    tmp_dir,
)

feature_dataset.upload()

feature_dataset.finalize()

# =====================================================
# Upload artifacts
# =====================================================

task.upload_artifact(
    "raw_dataset_id",
    raw_dataset_id,
)

task.upload_artifact(
    "feature_dataset_id",
    feature_dataset.id,
)

feature_dataset_info = {
    "dataset_id": feature_dataset.id,
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

task.get_logger().report_text(f"Feature Dataset ID = {feature_dataset.id}")

task.get_logger().report_text(f"Raw Dataset ID = {raw_dataset_id}")

task.get_logger().report_text(
    f"""
    Feature Dataset
    ------------------------
    id : {feature_dataset.id}

    train : {len(train_df)}

    valid : {len(valid_df)}

    test : {len(test_df)}
    """
)

print("Feature engineering completed.")

task.close()
