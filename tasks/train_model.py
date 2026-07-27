from clearml import (
    Task,
    OutputModel,
    Dataset,
)

from pathlib import Path

import pandas as pd

from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_TRAIN_NAME,
    RANDOM_STATE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SUPPORTED_MODELS,
)

# =====================================================
# Import Model Registry & Generic Training
# =====================================================

from business.models import (
    get_trainer_class,
    get_importance_calculator,
    get_model_config_class,
)
from business.train import (
    train_model,
    calculate_feature_importance,
    save_model,
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
        "hpo_task_id": "",  # Training Pipeline (single model)
        "compare_hpo_task_id": "",  # Model Selection Pipeline
        "model_type": "",  # Optional explicit model type
    }
)

# =====================================================
# Template creation mode
# =====================================================

if not params["feature_task_id"] or (not params["hpo_task_id"] and not params["compare_hpo_task_id"]):
    task.get_logger().report_text("Template creation mode.")
    task.close()
    raise SystemExit(0)

# =====================================================
# Determine model_type & hpo_task
# =====================================================

model_type = params.get("model_type", "").strip()  # Optional explicit param
best_model_type_from_compare = None

# Priority 1: Explicit model_type parameter
if model_type and model_type in SUPPORTED_MODELS:
    task.get_logger().report_text(f"📌 Using explicit model_type: {model_type}")
    hpo_task_id = params["compare_hpo_task_id"] or params["hpo_task_id"]

# Priority 2: Model Selection Pipeline - Auto-detect from compare task
elif params["compare_hpo_task_id"]:
    compare_hpo_task = Task.get_task(task_id=params["compare_hpo_task_id"])
    
    try:
        best_model_type_from_compare = wait_for_artifact(
            compare_hpo_task,
            "best_model_type",
            max_retries=10,
            wait_interval=2.0,
            logger_obj=task,
        )
        model_type = best_model_type_from_compare
        hpo_task_id = params["compare_hpo_task_id"]
        task.get_logger().report_text(f"✅ Auto-selected from Compare HPO: {model_type}")
    except Exception as e:
        task.get_logger().report_text(f"❌ Failed to get best_model_type: {str(e)}")
        task.close()
        raise SystemExit(1)

# Priority 3: Training Pipeline - Use single HPO task
elif params["hpo_task_id"]:
    hpo_task_id = params["hpo_task_id"]
    model_type = "lightgbm"  # Default for single HPO (backward compatibility)
    task.get_logger().report_text(f"📌 Training Pipeline mode: Using default {model_type}")

else:
    task.get_logger().report_text(
        "❌ No valid hpo_task_id or compare_hpo_task_id provided"
    )
    task.close()
    raise SystemExit(1)

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
# Load best params from HPO
# =====================================================

hpo_task = Task.get_task(task_id=hpo_task_id)

best_params = wait_for_artifact(
    hpo_task,
    "best_params",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

task.get_logger().report_text(f"Best params = {best_params}")


# =====================================================
# Get Model Trainer từ Registry
# =====================================================

task.get_logger().report_text(f\"📍 Initializing {model_type.upper()} trainer...\")

trainer_class = get_trainer_class(model_type)
config_class = get_model_config_class(model_type)

# Create config instance
config = config_class(random_state=RANDOM_STATE)

# Create trainer instance
trainer = trainer_class(config=config)

task.get_logger().report_text(f\"✅ {model_type.upper()} trainer initialized\")


# =====================================================
# Prepare Training Callbacks (model-specific)
# =====================================================

train_callbacks = None

if model_type == \"lightgbm\":
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
# Training info
# =====================================================

training_info = {
    "model_id": output_model.id,
    "model_type": model_type,
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
    "algorithm": model_type.upper(),
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
