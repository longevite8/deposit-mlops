from pathlib import Path

import pandas as pd
from clearml import Dataset, Task

from business.hpo import run_generic_hpo_optimization

# =====================================================
# Import Model Registry & Generic HPO
# =====================================================
from business.models import get_model_config_class, get_optimizer_class
from config import (
    FEATURE_COLUMNS,
    N_TRIALS,
    PROJECT_TEMPLATE,
    RANDOM_STATE,
    SUPPORTED_MODELS,
    TARGET_COLUMN,
    TEMPLATE_HPO_NAME,
)

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
        "model_type": "lightgbm",
    }
)


# =====================================================
# Validate Model Type
# =====================================================

if params["model_type"] not in SUPPORTED_MODELS:
    task.get_logger().report_text(
        f"❌ Unsupported model_type: {params['model_type']}. Supported: {SUPPORTED_MODELS}"
    )
    task.close()
    raise SystemExit(1)

task.get_logger().report_text(f"✅ Using model_type: {params['model_type']}")


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

from helpers import wait_for_artifact

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
# Definie Callback for Logging to ClearML
# =====================================================


def clearml_hpo_callback(study, trial):
    """
    Callback function chạy sau mỗi trial của Optuna.
    Gửi giá trị metric của trial hiện tại lên ClearML.
    """
    if trial.value is not None:
        task.get_logger().report_scalar(
            title="HPO Trials",
            series="Metric",
            value=trial.value,
            iteration=trial.number,
        )
        task.get_logger().report_text(
            f"Trial {trial.number} finished with value: {trial.value} and parameters: {trial.params}"
        )


# =====================================================
# Get Model Config & Optimizer từ Registry
# =====================================================

task.get_logger().report_text(f"📍 Initializing {params['model_type'].upper()} HPO...")

optimizer_class = get_optimizer_class(params["model_type"])
config_class = get_model_config_class(params["model_type"])

# Create config instance
config = config_class(random_state=RANDOM_STATE)

# Create optimizer instance
optimizer = optimizer_class(config=config)

task.get_logger().report_text(
    f"✅ {params['model_type'].upper()} optimizer initialized"
)

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

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
    "model_type": params["model_type"],
    "best_params": study.best_params,
    "best_score": study.best_value,
    "n_trials": N_TRIALS,
}
hpo_lineage = {
    "hpo_task_id": task.id,
    "feature_task_id": params["feature_task_id"],
    "feature_dataset_id": feature_dataset_id,
    "model_type": params["model_type"],
}
task.upload_artifact("hpo_summary", hpo_summary)
task.upload_artifact("hpo_lineage", hpo_lineage)


# =====================================================
# Scalars
# =====================================================

task.get_logger().report_single_value(
    "best_metric",
    float(best_trial.value),
)

task.get_logger().report_single_value(
    "n_trials",
    len(study.trials),
)

task.get_logger().report_text(f"Best params = {best_params}")

task.get_logger().report_text(f"Feature columns = {FEATURE_COLUMNS}")

task.get_logger().report_text(f"Model type = {params['model_type']}")


print(
    "Best params:",
    best_params,
)


task.flush()
task.close()
