import lightgbm as lgb
from clearml import (
    Task,
    OutputModel,
    Dataset,
)

from lightgbm import LGBMRegressor
from pathlib import Path

import pandas as pd
import joblib

from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_TRAIN_NAME,
    RANDOM_STATE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

from helpers import wait_for_artifact  # THÊM: Import từ helper

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_TRAIN_NAME,
    task_type=Task.TaskTypes.training,
)


# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "feature_task_id": "",
        "hpo_task_id": "",
    }
)

# =====================================================
# Template creation mode
# =====================================================

if not params["feature_task_id"] or not params["hpo_task_id"]:
    task.get_logger().report_text("Template creation mode.")

    task.close()
    raise SystemExit(0)


# =====================================================
# Load datasets
# =====================================================

feature_task = Task.get_task(
    task_id=params["feature_task_id"],
)

# SỬA: Lấy lineage của feature_task để truy xuất thông tin nguồn gốc
feature_lineage = wait_for_artifact(
    feature_task,
    "feature_lineage",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

feature_dataset_id = feature_lineage["feature_dataset_id"]

feature_dataset = Dataset.get(
    dataset_id=feature_dataset_id,
)

local_path = Path(feature_dataset.get_local_copy())

train_df = pd.read_parquet(local_path / "train.parquet")

valid_df = pd.read_parquet(local_path / "valid.parquet")

# =====================================================
# Combine train + valid
# =====================================================

df_train = pd.concat(
    [
        train_df,
        valid_df,
    ]
).reset_index(drop=True)


# =====================================================
# Dataset
# =====================================================

X_train = df_train[FEATURE_COLUMNS]

y_train = df_train[TARGET_COLUMN]

# Extract validation data từ valid_df
X_valid = valid_df[FEATURE_COLUMNS]

y_valid = valid_df[TARGET_COLUMN]

# =====================================================
# Load best params
# =====================================================

hpo_task = Task.get_task(task_id=params["hpo_task_id"])

best_params = hpo_task.artifacts["best_params"].get()

task.get_logger().report_text(f"Best params = {best_params}")


# =====================================================
# Train model
# =====================================================


# Callback để log loss mỗi epoch
def log_training_metrics(env):
    """Callback LightGBM để report scalars vào ClearML."""
    if env.iteration % 10 == 0:  # Log mỗi 10 iterations
        task.get_logger().report_scalar(
            title="Training Loss",
            series="iteration",
            value=env.evaluation_result_list[0][2],  # training loss
            iteration=env.iteration,
        )
        if len(env.evaluation_result_list) > 1:
            task.get_logger().report_scalar(
                title="Validation Loss",
                series="iteration",
                value=env.evaluation_result_list[1][2],  # validation loss
                iteration=env.iteration,
            )


callbacks = [
    lgb.log_evaluation(period=10),
    lgb.early_stopping(stopping_rounds=50),
    log_training_metrics,
]

model = LGBMRegressor(
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    num_leaves=best_params["num_leaves"],
    random_state=RANDOM_STATE,
    verbose=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=callbacks,
)


# =====================================================
# Save model
# =====================================================

joblib.dump(
    model,
    "model.pkl",
)


# =====================================================
# Register output model
# =====================================================

output_model = OutputModel(
    task=task,
    name="lightgbm_model",
)

output_model.update_weights(weights_filename="model.pkl")

output_model.set_metadata(
    "feature_dataset_id",
    feature_dataset_id,
)

output_model.set_metadata(
    "feature_task_id",
    feature_task.id,
)

output_model.set_metadata(
    "train_task_id",
    task.id,
)

# SỬA: Truy vết raw_dataset_id từ extract_task thông qua feature_lineage
extract_task_id = feature_lineage["extract_task_id"]
extract_task = Task.get_task(task_id=extract_task_id)
extract_summary = wait_for_artifact(
    extract_task,
    "extract_summary",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)
raw_dataset_id = extract_summary["raw_dataset_id"]

output_model.set_metadata(
    "raw_dataset_id",
    raw_dataset_id,
)

# =====================================================
# Upload model_id
# =====================================================

task.upload_artifact(
    "model_id",
    output_model.id,
)

task.upload_artifact(
    "feature_dataset_id",
    feature_dataset_id,
)

task.upload_artifact(
    "raw_dataset_id",
    raw_dataset_id,
)

# =====================================================
# Feature importance
# =====================================================

split_importance = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "split_importance": model.booster_.feature_importance(importance_type="split"),
    }
).sort_values(
    "split_importance",
    ascending=False,
)

gain_importance = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "gain_importance": model.booster_.feature_importance(importance_type="gain"),
    }
).sort_values(
    "gain_importance",
    ascending=False,
)

# Report feature importance scalars
task.get_logger().report_table(
    title="Feature Importance (Split)",
    series="importance",
    iteration=0,
    table_plot=split_importance,
)

task.get_logger().report_table(
    title="Feature Importance (Gain)",
    series="importance",
    iteration=0,
    table_plot=gain_importance,
)

# Log top 3 features
top_3_features = split_importance.head(3)
for idx, row in top_3_features.iterrows():
    task.get_logger().report_single_value(
        f"top_feature_{idx + 1}_importance",
        float(row["split_importance"]),
    )

task.upload_artifact(
    "split_importance",
    split_importance,
)

task.upload_artifact(
    "gain_importance",
    gain_importance,
)


# =====================================================
# Training info
# =====================================================

training_info = {
    "model_id": output_model.id,
    "feature_dataset_id": feature_dataset_id,
    "raw_dataset_id": raw_dataset_id,
    "best_params": best_params,
    "n_rows": len(df_train),
    "n_features": len(FEATURE_COLUMNS),
    "feature_columns": FEATURE_COLUMNS,
}

task.upload_artifact(
    "training_info",
    training_info,
)

model_card = {
    "algorithm": "LightGBM",
    "feature_columns": FEATURE_COLUMNS,
    "target_column": TARGET_COLUMN,
    "best_params": best_params,
    "n_rows": len(df_train),
    "n_features": len(FEATURE_COLUMNS),
    "feature_dataset_id": feature_dataset_id,
    "raw_dataset_id": raw_dataset_id,
}

task.upload_artifact(
    "model_card",
    model_card,
)

task.get_logger().report_text(
    f"""
    Raw Dataset      : {raw_dataset_id}
    Feature Dataset  : {feature_dataset_id}
    Training rows    : {len(df_train)}
    """
)

task.get_logger().report_text("Training completed.")

print("Training completed.")

# THÊM: Đồng bộ hoàn toàn trước khi kết thúc
task.flush()

# =====================================================
# Training summary & lineage
# =====================================================

train_summary = {
    "model_id": output_model.id,
    "model_name": output_model.name,
}
train_lineage = {
    "train_task_id": task.id,
    "feature_task_id": params["feature_task_id"],
    "hpo_task_id": params.get("hpo_task_id"),
    "model_id": output_model.id,
    "feature_dataset_id": feature_dataset_id,
}
task.upload_artifact("train_summary", train_summary)
task.upload_artifact("train_lineage", train_lineage)

task.close()
