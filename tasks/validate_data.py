from pathlib import Path
import time

import pandas as pd
from clearml import Dataset, Task

from config import (
    PROJECT_TEMPLATE,
    REQUIRED_COLUMNS,
    TEMPLATE_VALIDATE_NAME,
)

from helpers import wait_for_artifact
from business.validate import validate_data

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_VALIDATE_NAME,
    task_type=Task.TaskTypes.qc,
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
# Load datasets from feature Dataset (parquet)
# =====================================================

# Load feature dataset info
feature_task = Task.get_task(task_id=params["feature_task_id"])

feature_lineage = wait_for_artifact(
    feature_task,
    "feature_lineage",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)
feature_dataset_id = feature_lineage["feature_dataset_id"]

task.get_logger().report_text(f"📌 Loading feature dataset: {feature_dataset_id}")

# Một khoảng nghỉ cố định để đảm bảo ClearML Backend đã index xong Dataset từ step trước
time.sleep(5)

max_dataset_retries = 5
feature_dataset = None

for attempt in range(max_dataset_retries):
    try:
        # Thử lấy dataset instance
        feature_dataset = Dataset.get(dataset_id=feature_dataset_id)

        # Kiểm tra thêm thuộc tính project để đảm bảo SDK đã load đủ metadata
        if feature_dataset and feature_dataset.project:
            task.get_logger().report_text(
                f"✅ Feature dataset loaded: {feature_dataset.id}"
            )
            break
        else:
            raise ValueError("Dataset project metadata is not yet available.")

    except (Exception, TypeError) as e:
        # Bắt thêm TypeError vì SDK crash tại dòng check project name (NoneType)
        if attempt < max_dataset_retries - 1:
            task.get_logger().report_text(
                f"⚠️ Attempt {attempt + 1} failed to load metadata: {str(e)}. Retrying in 5s...",
                level="warning",
            )
            time.sleep(5)
        else:
            task.get_logger().report_text(
                f"⚠️ ID error after {max_dataset_retries} retries: {str(e)}. Trying by name fallback...",
                level="warning",
            )
            try:
                feature_dataset = Dataset.get(
                    dataset_name="Deposit Feature Dataset",
                )
            except Exception as final_e:
                task.get_logger().report_text(
                    f"❌ Final fallback failed: {str(final_e)}", level="error"
                )
                raise final_e

local_path = Path(feature_dataset.get_local_copy())

task.get_logger().report_text(f"📂 Dataset local path: {local_path}")

# Check file tồn tại không
train_parquet = local_path / "train.parquet"
if not train_parquet.exists():
    task.get_logger().report_text(
        f"❌ train.parquet not found in {local_path}\n"
        f"   Available files: {list(local_path.glob('*'))}",
        level="error",
    )
    raise FileNotFoundError(f"train.parquet not found in {local_path}")

train_df = pd.read_parquet(train_parquet)

task.get_logger().report_text(
    f"✅ Loaded training data: {len(train_df)} rows, {len(train_df.columns)} columns"
)

# =====================================================
# Validate on training set (representative split)
# =====================================================

df = train_df.copy().reset_index(drop=True)

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

validate_check = validate_data(df)

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

validation_report = {
    "schema_ok": validate_check["schema_check"]["schema_ok"],
    "missing_ok": validate_check["missing_check"]["missing_ok"],
    "dtype_ok": validate_check["dtype_check"]["dtype_ok"],
    "range_ok": validate_check["range_check"]["range_ok"],
    "passed": validate_check["passed"],
    "missing_columns": validate_check["schema_check"]["missing_columns"],
    "bad_columns": validate_check["dtype_check"]["bad_columns"],
    "n_rows": len(df),
    "n_features": len(REQUIRED_COLUMNS),
}

task.upload_artifact(
    "validation_report",
    validation_report,
)

# =====================================================
# Report scalars for observability
# =====================================================

task.get_logger().report_single_value("passed", int(validate_check["passed"]))

task.get_logger().report_single_value(
    "missing_rate_max", float(validate_check["missing_check"]["missing_rate_max"])
)

task.get_logger().report_single_value("n_rows", len(df))

summary_df = pd.DataFrame(
    [
        {
            "check": "schema",
            "passed": validate_check["schema_check"]["schema_ok"],
            "details": validate_check["schema_check"]["message"],
        },
        {
            "check": "missing_values",
            "passed": validate_check["missing_check"]["missing_ok"],
            "details": validate_check["missing_check"]["message"],
        },
        {
            "check": "dtypes",
            "passed": validate_check["dtype_check"]["dtype_ok"],
            "details": validate_check["dtype_check"]["message"],
        },
        {
            "check": "range",
            "passed": validate_check["range_check"]["range_ok"],
            "details": validate_check["range_check"]["message"],
        },
    ]
)

task.get_logger().report_table(
    title="Validation Report",
    series="checks",
    iteration=0,
    table_plot=summary_df,
)

if not validate_check["passed"]:
    raise ValueError(f"Validation failed: {validation_report}")

task.get_logger().report_text(str(validation_report))

print("✅ Validation passed.")

validate_summary = {
    "status": "PASS" if validate_check["passed"] else "FAIL",
    "checks_performed": summary_df.to_dict(orient="records"),
}
validate_lineage = {
    "validate_task_id": task.id,
    "feature_task_id": params["feature_task_id"],
    "feature_dataset_id": feature_dataset_id,
}
task.upload_artifact("validate_summary", validate_summary)
task.upload_artifact("validate_lineage", validate_lineage)

# =====================================================
# Final flush before closing
# =====================================================

task.flush()
task.close()
