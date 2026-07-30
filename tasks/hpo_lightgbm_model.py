"""
HPO LightGBM - Hyperparameter Optimization cho LightGBM Model.
"""

from pathlib import Path

import pandas as pd
from clearml import Dataset, Task

from business.hpo import run_generic_hpo_optimization
from business.models import get_model_config_class, get_optimizer_class
from config import (
    FEATURE_COLUMNS,
    FORECAST_HORIZON,
    N_TRIALS,
    PROJECT_TEMPLATE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEMPLATE_HPO_LIGHTGBM_NAME,
)
from helpers import wait_for_artifact

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_HPO_LIGHTGBM_NAME,
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
# Multi-target extraction (for multi-target strategy)
# =====================================================

# Check if multi-step targets exist (created by create_multistep_targets)
target_cols = [col for col in train_df.columns if col.startswith("target_")]
if target_cols:
    # Use multi-target columns for LightGBM multi-output training
    y_train = train_df[target_cols]
    y_valid = valid_df[target_cols]
    task.get_logger().report_text(
        f"✅ Using multi-target strategy with {len(target_cols)} targets: {target_cols}"
    )
else:
    # Fallback to single target if multi-targets not available
    task.get_logger().report_text(
        "⚠️ No multi-target columns found, using single target"
    )

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

task.get_logger().report_text("📍 Starting LightGBM HPO...")

optimizer_class = get_optimizer_class("lightgbm")
config_class = get_model_config_class("lightgbm")

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
# BUSINESS LOGIC: End
# =====================================================

best_params = study.best_params
best_score = study.best_value

# =====================================================
# Upload Artifacts
# =====================================================

task.upload_artifact("best_params", best_params)
task.upload_artifact("best_score", best_score)
task.upload_artifact("model_type", "lightgbm")

task.get_logger().report_single_value("best_score", float(best_score))
task.get_logger().report_text(
    f"✅ LightGBM HPO Completed\n"
    f"   Best Score: {best_score:.6f}\n"
    f"   Best Params: {best_params}"
)

task.flush()
task.close()
