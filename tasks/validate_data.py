from pathlib import Path

import pandas as pd
from clearml import Dataset, Task

from config import (
    TARGET_COLUMN,
    MIN_VALUE,
    MAX_VALUE,
    MAX_MISSING_RATE,
    PROJECT_TEMPLATE,
    REQUIRED_COLUMNS,
    TEMPLATE_VALIDATE_NAME,
)

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

# SỬA: Lấy giá trị artifact một cách an toàn
feature_dataset_id = feature_task.artifacts["feature_dataset_id"].get()

# THÊM: Log debug để kiểm tra ID trên Web UI
print(f"DEBUG: Feature Dataset ID retrieved: {feature_dataset_id}")

# SỬA: Bọc Dataset.get để tránh crash nếu ID chưa sẵn sàng
try:
    feature_dataset = Dataset.get(dataset_id=feature_dataset_id)
except Exception as e:
    task.get_logger().report_text(
        f"Error: {feature_dataset_id} is not a valid Dataset ID yet."
    )
    raise ValueError(
        f"ID {feature_dataset_id} is not a finalized Dataset. Check Feature Task status."
    ) from e

local_path = Path(feature_dataset.get_local_copy())

train_df = pd.read_parquet(local_path / "train.parquet")

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

print("Validation passed.")

task.close()
