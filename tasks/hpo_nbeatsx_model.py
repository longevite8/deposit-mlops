"""
HPO NBEATSx - Hyperparameter Optimization cho NBEATSx Model.
"""

import math
from pathlib import Path

import pandas as pd
from clearml import Dataset, Task
from optuna.trial import TrialState

from business.hpo import run_generic_hpo_optimization
from business.models import get_model_config_class, get_optimizer_class
from config import (
    FEATURE_COLUMNS,
    FORECAST_HORIZON,
    N_TRIALS,
    PROJECT_TEMPLATE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEMPLATE_HPO_NBEATSX_NAME,
)
from helpers import wait_for_artifact

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_HPO_NBEATSX_NAME,
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
# Load Data
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

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]
X_valid = valid_df[FEATURE_COLUMNS]
y_valid = valid_df[TARGET_COLUMN]

# =====================================================
# Callback
# =====================================================


def clearml_hpo_callback(study, trial):
    if trial.value is not None:
        task.get_logger().report_scalar(
            title="HPO Trials", series="MAPE", value=trial.value, iteration=trial.number
        )


# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

task.get_logger().report_text("📍 Starting NBEATSx HPO...")

optimizer_class = get_optimizer_class("nbeatsx")
config_class = get_model_config_class("nbeatsx")

config = config_class(random_state=RANDOM_STATE, forecast_horizon=FORECAST_HORIZON)
optimizer = optimizer_class(config=config)

study = run_generic_hpo_optimization(
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
    callbacks=[clearml_hpo_callback],
)

# =====================================================
# Validate HPO Results
# =====================================================

completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]

if not completed_trials:
    error_msg = (
        f"❌ NBEATSx HPO failed: không có trial nào thành công\n"
        f"   Completed: {len(completed_trials)}\n"
        f"   Failed: {len(failed_trials)}"
    )
    task.get_logger().report_text(error_msg)
    raise RuntimeError(error_msg)

best_params = study.best_params
best_score = study.best_value

if best_score is None or not math.isfinite(float(best_score)):
    error_msg = f"❌ NBEATSx HPO failed: invalid best_score={best_score}"
    task.get_logger().report_text(error_msg)
    raise RuntimeError(error_msg)

# =====================================================
# Upload Artifacts
# =====================================================

hpo_nbeatsx_summary = {
    "best_params": best_params,
    "best_score": best_score,
    "model_type": "nbeatsx",
    "n_completed_trials": len(completed_trials),
    "n_failed_trials": len(failed_trials),
}

hpo_nbeatsx_lineage = {
    "hpo_nbeatsx_task_id": task.id,
    "feature_task_id": params["feature_task_id"],
    "feature_dataset_id": feature_lineage["feature_dataset_id"],
}

task.upload_artifact("hpo_nbeatsx_summary", hpo_nbeatsx_summary)
task.upload_artifact("hpo_nbeatsx_lineage", hpo_nbeatsx_lineage)

task.get_logger().report_single_value("best_score", float(best_score))
task.get_logger().report_text(
    f"✅ NBEATSx HPO Completed\n"
    f"   Completed Trials: {len(completed_trials)}\n"
    f"   Failed Trials: {len(failed_trials)}\n"
    f"   Best Score: {best_score:.6f}\n"
    f"   Best Params: {best_params}"
)

task.flush()
task.close()
