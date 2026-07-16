import tempfile
import os
import time  # Đảm bảo đã import
from clearml import (
    Task,
    Dataset,
)
from config import (
    PROJECT_TEMPLATE,
    PROJECT_DATASET,
    TEMPLATE_EXTRACT_NAME,
)
from business.data_processing import (
    build_extract_source_summary,
    extract_data,
    get_source_config,
)


task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_EXTRACT_NAME,
    task_type=Task.TaskTypes.data_processing,
)

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

source_config = get_source_config()
df = extract_data(source_config)

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

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
# QUAN TRỌNG: Flush task trước khi upload artifact
# =====================================================

task.flush()

# THÊM: Chờ server đồng bộ Task trước khi tạo Dataset
time.sleep(3)

# =====================================================
# Build Raw Dataset (với retry logic)
# =====================================================

max_retries = 5
dataset = None

for attempt in range(max_retries):
    try:
        task.get_logger().report_text(
            f"📌 Creating raw dataset (attempt {attempt + 1}/{max_retries})..."
        )

        dataset = Dataset.create(
            dataset_project=PROJECT_DATASET,
            dataset_name="Deposit Raw Dataset",
        )

        dataset.add_files(tmp_dir)
        dataset.upload()
        dataset.finalize()

        task.get_logger().report_text(f"✅ Created new dataset: {dataset.id}")
        break  # Thành công thì thoát vòng lặp

    except Exception as e:
        task.get_logger().report_text(
            f"⚠️ Attempt {attempt + 1} failed: {str(e)}",
            level="warning",
        )

        if attempt < max_retries - 1:
            wait_sec = (attempt + 1) * 3  # Backoff: 3s, 6s, 9s, 12s
            task.get_logger().report_text(
                f"   Retrying in {wait_sec}s...",
                level="warning",
            )
            time.sleep(wait_sec)
        else:
            # Hết retry mới fallback sang Dataset.get()
            task.get_logger().report_text(
                f"⚠️ Cannot create new dataset after {max_retries} attempts.\n"
                f"   Falling back to latest finalized version...",
                level="warning",
            )
            try:
                dataset = Dataset.get(
                    dataset_project=PROJECT_DATASET,
                    dataset_name="Deposit Raw Dataset",
                )
                task.get_logger().report_text(
                    f"✅ Using existing dataset: {dataset.id}"
                )
            except Exception as final_e:
                task.get_logger().report_text(
                    f"❌ Final fallback failed: {str(final_e)}", level="error"
                )
                raise final_e

if not dataset:
    raise ValueError("Could not create or get raw dataset after all retries.")

# =====================================================
# Upload artifact
# =====================================================

extract_summary = {
    "n_rows": len(df),
    "raw_dataset_id": dataset.id,
    **build_extract_source_summary(source_config, df),
}
extract_lineage = {
    "extract_task_id": task.id,
    "raw_dataset_id": dataset.id,
    "source_type": "postgresql",
    "source_project_name": source_config.project_name,
    "source_from_date": source_config.from_date,
    "source_to_date": source_config.to_date,
}
task.upload_artifact("extract_summary", extract_summary)
task.upload_artifact("extract_lineage", extract_lineage)
task.upload_artifact("raw_data", df)

task.get_logger().report_text(f"raw_dataset_id={dataset.id}")

print(df.head())
print("✅ Extract completed.")

task.flush()
task.close()
