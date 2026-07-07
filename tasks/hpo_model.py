import optuna
from pathlib import Path

import pandas as pd
from clearml import Dataset, Task
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    r2_score,
)

from config import (
    FEATURE_COLUMNS,
    N_TRIALS,
    PROJECT_TEMPLATE,
    RANDOM_STATE,
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

feature_dataset_id = feature_task.artifacts["feature_dataset_id"].get()

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
# Objective
# =====================================================


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: train LGBMRegressor and return validation MAPE."""
    model_params = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.1,
            log=True,
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves",
            15,
            100,
        ),
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            1000,
        ),
        "random_state": RANDOM_STATE,
    }

    model = LGBMRegressor(**model_params)

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(X_valid)

    mape = mean_absolute_percentage_error(
        y_valid,
        y_pred,
    )

    r2 = r2_score(
        y_valid,
        y_pred,
    )

    task.get_logger().report_scalar(
        title="MAPE",
        series="trial",
        value=mape,
        iteration=trial.number,
    )

    task.get_logger().report_scalar(
        title="R2",
        series="trial",
        value=r2,
        iteration=trial.number,
    )

    return mape


# =====================================================
# Run study
# =====================================================

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
)


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
    "best_mape": float(best_trial.value),
    "n_trials": len(study.trials),
    "best_trial_number": best_trial.number,
}

task.upload_artifact(
    "hpo_summary",
    hpo_summary,
)

hpo_lineage = {
    "hpo_task_id": task.id,
    "feature_task_id": feature_task.id,
    "feature_dataset_id": feature_dataset_id,
}

task.upload_artifact(
    "hpo_lineage",
    hpo_lineage,
)


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


task.close()
