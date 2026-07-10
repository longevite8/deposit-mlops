from pathlib import Path

import pandas as pd
from clearml import Dataset, Task

from config import (
    TARGET_COLUMN,
    MIN_VALUE,
    MAX_VALUE,
    MAX_MISSING_RATE,
    PROJECT_TEMPLATE,
    PROJECT_DATASET,
    REQUIRED_COLUMNS,
    TEMPLATE_VALIDATE_NAME,
)

from helpers import wait_for_artifact  # THÊM: Import từ helper

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

feature_task = Task.get_task(task_id=params["feature_task_id"])

# SỬA: Dùng wait_for_artifact để chắc chắn dataset ID sẵn sàng
feature_dataset_id = wait_for_artifact(
    feature_task,
    "feature_dataset_id",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

task.get_logger().report_text(f"📌 Loading feature dataset: {feature_dataset_id}")

# SỬA: Thêm try-except với fallback logic
try:
    feature_dataset = Dataset.get(dataset_id=feature_dataset_id)
    task.get_logger().report_text(
        f"✅ Feature dataset loaded by ID: {feature_dataset.id}"
    )
except ValueError as e:
    task.get_logger().report_text(
        f"⚠️ Cannot load dataset by ID ({feature_dataset_id}): {str(e)}\n"
        f"   Falling back to latest finalized by name...",
        level="warning",
    )

    try:
        feature_dataset = Dataset.get(
            dataset_project=PROJECT_DATASET,
            dataset_name="Deposit Feature Dataset",
        )
        task.get_logger().report_text(
            f"✅ Feature dataset loaded by name (fallback): {feature_dataset.id}"
        )
    except Exception as e2:
        task.get_logger().report_text(
            f"❌ Failed to load feature dataset:\n"
            f"   By ID: {str(e)}\n"
            f"   By Name: {str(e2)}",
            level="error",
        )
        raise ValueError(f"Could not load feature dataset: {e2}")

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
# Schema validation
# =====================================================

missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

schema_ok = len(missing_columns) == 0

# =====================================================
# Missing value validation
# =====================================================

missing_rate = df[REQUIRED_COLUMNS].isna().mean()

missing_ok = missing_rate.max() <= MAX_MISSING_RATE

# =====================================================
# Dtype validation
# =====================================================

bad_columns = []

for col in REQUIRED_COLUMNS:
    if str(df[col].dtype) not in (
        "int64",
        "float64",
        "bool",
    ):
        bad_columns.append(col)

dtype_ok = len(bad_columns) == 0

# =====================================================
# Range validation
# =====================================================

range_ok = df[TARGET_COLUMN].between(MIN_VALUE, MAX_VALUE).all()

if not range_ok:
    print(
        f"❌ Range validation failed: {TARGET_COLUMN} contains values outside [{MIN_VALUE}, {MAX_VALUE}]"
    )
    task.get_logger().report_text(f"Range validation failed for {TARGET_COLUMN}")
    task.close()
    raise SystemExit(1)

print("✅ Range validation passed.")

# =====================================================
# Final result
# =====================================================

passed = schema_ok and missing_ok and dtype_ok and range_ok

validation_report = {
    "schema_ok": schema_ok,
    "missing_ok": missing_ok,
    "dtype_ok": dtype_ok,
    "range_ok": range_ok,
    "passed": passed,
    "missing_columns": missing_columns,
    "bad_columns": bad_columns,
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

task.get_logger().report_single_value("passed", int(passed))

task.get_logger().report_single_value("missing_rate_max", float(missing_rate.max()))

task.get_logger().report_single_value("n_rows", len(df))

summary_df = pd.DataFrame(
    [
        {
            "check": "schema",
            "passed": schema_ok,
            "details": str(missing_columns) if missing_columns else "OK",
        },
        {
            "check": "missing_values",
            "passed": missing_ok,
            "details": f"max_rate={missing_rate.max():.4f}",
        },
        {
            "check": "dtypes",
            "passed": dtype_ok,
            "details": str(bad_columns) if bad_columns else "OK",
        },
        {
            "check": "range",
            "passed": range_ok,
            "details": "target > 0",
        },
    ]
)

task.get_logger().report_table(
    title="Validation Report",
    series="checks",
    iteration=0,
    table_plot=summary_df,
)

if not passed:
    raise ValueError(f"Validation failed: {validation_report}")

task.get_logger().report_text(str(validation_report))

print("✅ Validation passed.")

# =====================================================
# Final flush before closing
# =====================================================

task.flush()

validate_summary = {
    "status": "PASS" if passed else "FAIL",
    "checks_performed": len(REQUIRED_COLUMNS),
    "failed_checks": len(bad_columns) + (1 if not range_ok else 0),
}
validate_lineage = {
    "validate_task_id": task.id,
    "feature_task_id": params[
        "feature_task_id"
    ],  # Sửa từ extract_task_id thành feature_task_id
}
task.upload_artifact("validate_summary", validate_summary)
task.upload_artifact("validate_lineage", validate_lineage)

task.close()
