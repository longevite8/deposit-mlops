from pathlib import Path

import pandas as pd
from clearml import Dataset, Task

from config import (
    FEATURE_COLUMNS,
    N_TRIALS,
    PROJECT_TEMPLATE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEMPLATE_HPO_NAME,
)

from business.hpo import run_hpo_optimization
from helpers import wait_for_artifact

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_HPO_NAME,
    task_type=Task.TaskTypes.optimizer,
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

feature_lineage = wait_for_artifact(
    feature_task,
    "feature_lineage",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

feature_dataset_id = feature_lineage["feature_dataset_id"]

feature_dataset = Dataset.get(dataset_id=feature_dataset_id)

local_path = Path(feature_dataset.get_local_copy())

train_df = pd.read_parquet(local_path / "train.parquet")

valid_df = pd.read_parquet(local_path / "valid.parquet")


# =====================================================
# Dataset
# =====================================================

X_train = train_df[FEATURE_COLUMNS]

y_train = train_df[TARGET_COLUMN]

X_valid = valid_df[FEATURE_COLUMNS]

y_valid = valid_df[TARGET_COLUMN]

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

study = run_hpo_optimization(
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
)

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

# =====================================================
# Best params
# =====================================================

best_params = study.best_params

best_trial = study.best_trial

# =====================================================
# Artifacts
# =====================================================

task.upload_artifact(
    name="best_params",
    artifact_object=best_params,
)

hpo_summary = {
    "best_params": study.best_params,
    "best_score": study.best_value,
    "n_trials": N_TRIALS,
}
hpo_lineage = {
    "hpo_task_id": task.id,
    "feature_task_id": params["feature_task_id"],
    "feature_dataset_id": feature_dataset_id,
}
task.upload_artifact("hpo_summary", hpo_summary)
task.upload_artifact("hpo_lineage", hpo_lineage)


# =====================================================
# Scalars
# =====================================================

task.get_logger().report_single_value(
    "best_mape",
    float(best_trial.value),
)

task.get_logger().report_single_value(
    "n_trials",
    len(study.trials),
)

task.get_logger().report_text(f"Best params = {best_params}")

task.get_logger().report_text(f"Feature columns = {FEATURE_COLUMNS}")


print(
    "Best params:",
    best_params,
)


task.flush()
task.close()
