from pathlib import Path

import pandas as pd
from clearml import (
    Dataset,
    OutputModel,
    Task,
)

# =====================================================
# Import Model Registry & Generic Training
# =====================================================
from business.models import (
    get_importance_calculator,
    get_model_config_class,
    get_trainer_class,
)
from business.train import (
    calculate_feature_importance,
    save_model,
    train_model,
)
from config import (
    FEATURE_COLUMNS,
    FORECAST_HORIZON,
    PROJECT_TEMPLATE,
    RANDOM_STATE,
    SUPPORTED_MODELS,
    TARGET_COLUMN,
    TEMPLATE_TRAIN_NAME,
)
from helpers import wait_for_artifact

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
        "compare_hpo_task_id": "",
    }
)

# =====================================================
# Template creation mode
# =====================================================

if not params["feature_task_id"] or not params["compare_hpo_task_id"]:
    task.get_logger().report_text("Template creation mode.")
    task.close()
    raise SystemExit(0)

# =====================================================
# Determine model_type & compare_hpo task
# =====================================================

compare_hpo_task = Task.get_task(task_id=params["compare_hpo_task_id"])
compare_hpo_summary = wait_for_artifact(
    compare_hpo_task,
    "compare_hpo_summary",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)
model_type = compare_hpo_summary["best_model_type"]
compare_hpo_task_id = params["compare_hpo_task_id"]
task.get_logger().report_text(
    f"✅ Auto-selected from Compare HPO: {model_type} "
    f"(score: {compare_hpo_summary['best_score']:.6f})"
)


# Validate model_type
if model_type not in SUPPORTED_MODELS:
    task.get_logger().report_text(
        f"❌ Invalid model_type: {model_type}. Supported: {SUPPORTED_MODELS}"
    )
    task.close()
    raise SystemExit(1)

task.get_logger().report_text(f"✅ Model type: {model_type}")


# =====================================================
# Load datasets
# =====================================================

feature_task = Task.get_task(
    task_id=params["feature_task_id"],
)

# Lấy lineage của feature_task để truy xuất thông tin nguồn gốc
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
# Multi-target extraction (Model-Aware Strategy)
# =====================================================

# Check if multi-step targets exist (created by create_multistep_targets)
target_cols = [col for col in df_train.columns if col.startswith("target_")]

if target_cols and model_type in ["nbeatsx", "nhits"]:
    # ✅ Neural models HỖTRỢ multi-target (multi-step forecasting)
    y_train = df_train[target_cols]
    y_valid = valid_df[target_cols]
    task.get_logger().report_text(
        f"✅ Neural model {model_type.upper()} using multi-target strategy "
        f"with {len(target_cols)} targets: {target_cols}"
    )
elif target_cols and model_type == "lightgbm":
    # ❌ LightGBM KHÔNG hỗ trợ multi-target → chỉ dùng TARGET_COLUMN (single-step)
    y_train = df_train[TARGET_COLUMN]
    y_valid = valid_df[TARGET_COLUMN]
    task.get_logger().report_text(
        f"⚠️ LightGBM không hỗ trợ multi-target, "
        f"chuyển sang single-target strategy: {TARGET_COLUMN}"
    )
else:
    # Fallback to single target if multi-targets not available
    y_train = df_train[TARGET_COLUMN]
    y_valid = valid_df[TARGET_COLUMN]
    task.get_logger().report_text(f"✅ Using single-target strategy: {TARGET_COLUMN}")

# =====================================================
# Get best params from HPO
# =====================================================

best_params = compare_hpo_summary.get("best_params")

if not isinstance(best_params, dict) or not best_params:
    task.get_logger().report_text(
        f"❌ Missing best_params for selected model: {model_type}"
    )
    task.close(status="failed")
    raise SystemExit(1)

task.get_logger().report_text(f"Best params = {best_params}")


# =====================================================
# Get Model Trainer từ Registry
# =====================================================

task.get_logger().report_text(f"📍 Initializing {model_type.upper()} trainer...")

trainer_class = get_trainer_class(model_type)
config_class = get_model_config_class(model_type)

# Create config instance with forecast horizon
config = config_class(random_state=RANDOM_STATE, forecast_horizon=FORECAST_HORIZON)

# Create trainer instance
trainer = trainer_class(config=config)

task.get_logger().report_text(f"✅ {model_type.upper()} trainer initialized")


# =====================================================
# Prepare Training Callbacks (model-specific)
# =====================================================

train_callbacks = None

if model_type == "lightgbm":
    import lightgbm as lgb

    def log_training_metrics(env):
        if env.iteration % 10 == 0:
            task.get_logger().report_scalar(
                title="Training Loss",
                series="iteration",
                value=env.evaluation_result_list[0][2],
                iteration=env.iteration,
            )
            if len(env.evaluation_result_list) > 1:
                task.get_logger().report_scalar(
                    title="Validation Loss",
                    series="iteration",
                    value=env.evaluation_result_list[1][2],
                    iteration=env.iteration,
                )

    train_callbacks = [
        lgb.log_evaluation(period=10),
        lgb.early_stopping(stopping_rounds=50),
        log_training_metrics,
    ]

elif model_type in ["nbeatsx", "nhits"]:
    # Neural models sử dụng PyTorch Lightning callbacks
    # Callbacks được setup trong model trainer class
    train_callbacks = None

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

model = train_model(
    trainer=trainer,
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    best_params=best_params,
    callbacks=train_callbacks,
)

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

# =====================================================
# Calculate Feature Importance (if supported)
# =====================================================

importance_calculator_class = get_importance_calculator(model_type)

split_importance = None
gain_importance = None

if importance_calculator_class is not None:
    # Model supports feature importance
    task.get_logger().report_text("📍 Calculating feature importance...")

    importance_calculator = importance_calculator_class()

    split_importance, gain_importance = calculate_feature_importance(
        calculator=importance_calculator,
        model=model,
        feature_names=FEATURE_COLUMNS,
    )

    task.get_logger().report_text("✅ Feature importance calculated")

    # Upload artifacts
    task.upload_artifact("split_importance", split_importance)
    if gain_importance is not None:
        task.upload_artifact("gain_importance", gain_importance)

    # Report feature importance to UI
    task.get_logger().report_table(
        title="Feature Importance (Primary)",
        series="importance",
        iteration=0,
        table_plot=split_importance,
    )

    if gain_importance is not None:
        task.get_logger().report_table(
            title="Feature Importance (Secondary)",
            series="importance",
            iteration=0,
            table_plot=gain_importance,
        )

    # Log top 3 features
    top_3_features = split_importance.head(3)
    for idx, row in top_3_features.iterrows():
        feature_name = row.iloc[0]  # First column is feature name
        importance_value = row.iloc[1]  # Second column is importance value
        task.get_logger().report_single_value(
            f"top_feature_{idx + 1}_importance",
            float(importance_value),
        )

else:
    # Model doesn't support feature importance
    task.get_logger().report_text(
        f"⚠️ {model_type.upper()} is a neural model and doesn't support feature importance."
    )


# =====================================================
# Save Model
# =====================================================

model_path = save_model(model, "model.pkl")
task.get_logger().report_text(f"✅ Model saved to {model_path}")


# =====================================================
# Register output model
# =====================================================

output_model = OutputModel(
    task=task,
    name=f"{model_type}_model",
)

output_model.update_weights(weights_filename="model.pkl")

# ✅ UPLOAD model_id artifact explicitly
# This is what the evaluate task will look for
task.get_logger().report_text(f"📍 Uploading model_id artifact: {output_model.id}")
task.upload_artifact("model_id", output_model.id)

# ✅ UPLOAD model_type artifact explicitly
# This is what the evaluate task will look for
task.get_logger().report_text(f"📍 Uploading model_type artifact: {model_type}")
task.upload_artifact("model_type", model_type)

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

output_model.set_metadata(
    "model_type",
    model_type,
)

# Truy vết raw_dataset_id từ extract_task thông qua feature_lineage
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


task.upload_artifact(
    "feature_dataset_id",
    feature_dataset_id,
)

task.upload_artifact(
    "raw_dataset_id",
    raw_dataset_id,
)

# ✅ UPLOAD compare_hpo_task_id artifacts for downstream tasks
task.get_logger().report_text("📍 Uploading compare_hpo_task_id artifacts...")
task.upload_artifact("compare_hpo_task_id", params["compare_hpo_task_id"])
task.get_logger().report_text(
    f"   compare_hpo_task_id: {params['compare_hpo_task_id']}"
)


# =====================================================
# Training Summary & Lineage
# =====================================================

training_summary = {
    "model_id": output_model.id,
    "model_type": model_type,
    "best_params": best_params,
    "n_rows": len(df_train),
    "n_features": len(FEATURE_COLUMNS),
    "feature_columns": FEATURE_COLUMNS,
}

training_lineage = {
    "train_task_id": task.id,
    "feature_task_id": params["feature_task_id"],
    "feature_dataset_id": feature_dataset_id,
    "compare_hpo_task_id": params["compare_hpo_task_id"],
    "raw_dataset_id": raw_dataset_id,
}

task.upload_artifact("training_summary", training_summary)
task.upload_artifact("training_lineage", training_lineage)


# =====================================================
# Scalars
# =====================================================

task.get_logger().report_single_value("n_rows", len(df_train))
task.get_logger().report_single_value("n_features", len(FEATURE_COLUMNS))
task.get_logger().report_text(
    f"Training completed successfully! Model type: {model_type}"
)


task.flush()
task.close()
