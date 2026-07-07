import numpy as np
import pandas as pd
import tempfile
from clearml import (
    Task,
    Dataset,
)
from config import (
    PROJECT_TEMPLATE,
    PROJECT_DATASET,
    TEMPLATE_EXTRACT_NAME,
    RANDOM_STATE,
    DATE_COLUMN,
    TARGET_COLUMN,  # ← THAY: RAW_TARGET_COLUMN → TARGET_COLUMN
    START_DATE,
    N_DAYS,
    GAMMA_SHAPE,
    GAMMA_SCALE,
)
import os

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_EXTRACT_NAME,
    task_type=Task.TaskTypes.data_processing,
)


# =====================================================
# Reproducibility
# =====================================================

np.random.seed(RANDOM_STATE)


# =====================================================
# Generate synthetic data
# =====================================================

dates = pd.date_range(
    start=START_DATE,
    periods=N_DAYS,
)

df = pd.DataFrame(
    {
        DATE_COLUMN: dates,
        TARGET_COLUMN: np.random.gamma(
            shape=GAMMA_SHAPE,
            scale=GAMMA_SCALE,
            size=len(dates),
        ),
    }
)

task.upload_artifact(
    "raw_data",
    df,
)

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

# =====================================================
# Build Raw Dataset
# =====================================================

dataset = Dataset.create(
    dataset_project=PROJECT_DATASET,
    dataset_name="Deposit Raw Dataset",
)

dataset.add_files(tmp_dir)
dataset.upload()
dataset.finalize()

task.upload_artifact(
    "raw_dataset_id",
    dataset.id,
)

task.get_logger().report_text(f"raw_dataset_id={dataset.id}")

print(df.head())
print("✅ Extract completed.")

task.close()
