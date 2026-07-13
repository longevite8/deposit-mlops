import tempfile
import os
from clearml import (
    Task,
    Dataset,
)
from config import (
    PROJECT_TEMPLATE,
    PROJECT_DATASET,
    TEMPLATE_EXTRACT_NAME,
    START_DATE,
    N_DAYS,
)
from business.data_processing import extract_data


task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_EXTRACT_NAME,
    task_type=Task.TaskTypes.data_processing,
)

# =====================================================
# BUSINESS LOGIC
# =====================================================

df = extract_data()


# =====================================================
# Convert df to parquet file for storing in ClearML Dataset
# =====================================================

tmp_dir = tempfile.mkdtemp()

raw_file = os.path.join(
    tmp_dir,
    "raw.parquet",
)

df.to_parquet(
    raw_file,
    index=False,
)

# Đảm bảo Task hiện tại đã đồng bộ hoàn toàn với Server trước khi tạo Dataset
task.flush()

# =====================================================
# Build Raw Dataset
# =====================================================

try:
    # Thử tạo mới Dataset
    dataset = Dataset.create(
        dataset_project=PROJECT_DATASET,
        dataset_name="Deposit Raw Dataset",
    )

    dataset.add_files(tmp_dir)
    dataset.upload()
    dataset.finalize()

    task.get_logger().report_text(f"✅ Created new dataset: {dataset.id}")

except Exception as e:
    # Nếu lỗi (Dataset đã tồn tại), lấy version finalized mới nhất theo tên
    task.get_logger().report_text(
        f"⚠️ Cannot create new dataset: {str(e)}. Falling back to latest finalized version."
    )

    dataset = Dataset.get(
        dataset_project=PROJECT_DATASET,
        dataset_name="Deposit Raw Dataset",
    )

    task.get_logger().report_text(f"✅ Using existing dataset: {dataset.id}")

# =====================================================
# QUAN TRỌNG: Flush task trước khi upload artifact
# =====================================================

task.flush()

# =====================================================
# Upload artifact
# =====================================================

extract_summary = {
    "n_rows": len(df),
    "start_date": START_DATE,
    "n_days": N_DAYS,
    "raw_dataset_id": dataset.id,
}
extract_lineage = {
    "extract_task_id": task.id,
    "raw_dataset_id": dataset.id,
}
task.upload_artifact("extract_summary", extract_summary)
task.upload_artifact("extract_lineage", extract_lineage)
task.upload_artifact("raw_data", df)

task.get_logger().report_text(f"raw_dataset_id={dataset.id}")

print(df.head())
print("✅ Extract completed.")

task.flush()
task.close()
