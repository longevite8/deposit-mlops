from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import os

import pandas as pd
from clearml import Dataset, OutputModel, Task

from business.forecasting import (
    ForecastTrainingConfig,
    archive_model_dir,
    historical_exogenous_columns,
    prepare_forecast_frame,
    save_forecast_model,
    train_forecast_model,
)
from config import (
    FORECAST_ACTIVATION,
    FORECAST_CV_WINDOWS,
    FORECAST_EVAL_HORIZON,
    FORECAST_HORIZON,
    FORECAST_HORIZON_METADATA_KEY,
    FORECAST_INPUT_SIZE,
    FORECAST_LEARNING_RATE,
    FORECAST_LOSS,
    FORECAST_MAX_STEPS,
    FORECAST_OPTIMIZER,
    FORECAST_OUTPUT_DIR,
    FORECAST_START_PADDING_ENABLED,
    FORECAST_UNIQUE_ID,
    PROJECT_TEMPLATE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEMPLATE_TRAIN_NAME,
)
from helpers import wait_for_artifact


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_TRAIN_NAME,
    task_type=Task.TaskTypes.training,
)


params = task.connect(
    {
        "feature_task_id": "",
        "hpo_task_id": "",
        "horizon": FORECAST_HORIZON,
        "eval_horizon": FORECAST_EVAL_HORIZON,
        "input_size": FORECAST_INPUT_SIZE,
        "learning_rate": FORECAST_LEARNING_RATE,
        "max_steps": FORECAST_MAX_STEPS,
        "loss": FORECAST_LOSS,
        "optimizer": FORECAST_OPTIMIZER,
        "activation": FORECAST_ACTIVATION,
        "cv_windows": FORECAST_CV_WINDOWS,
        "start_padding_enabled": FORECAST_START_PADDING_ENABLED,
        "output_dir": FORECAST_OUTPUT_DIR,
        "unique_id": FORECAST_UNIQUE_ID,
    }
)


if not params["feature_task_id"]:
    task.get_logger().report_text("Template creation mode.")
    task.close()
    raise SystemExit(0)


forecast_config = ForecastTrainingConfig(
    horizon=int(params["horizon"]),
    input_size=int(params["input_size"]),
    learning_rate=float(params["learning_rate"]),
    max_steps=int(params["max_steps"]),
    loss=str(params["loss"]),
    optimizer=str(params["optimizer"]),
    activation=str(params["activation"]),
    cv_windows=int(params["cv_windows"]),
    eval_horizon=int(params["eval_horizon"]),
    seed=RANDOM_STATE,
    start_padding_enabled=truthy(params["start_padding_enabled"]),
)
output_dir = Path(str(params["output_dir"]))
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = output_dir / "checkpoints"
model_archive_path = output_dir / "deposit_forecast_model.zip"


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

df_train = pd.concat([train_df, valid_df], ignore_index=True)
forecast_train_df = prepare_forecast_frame(
    df_train,
    unique_id=str(params["unique_id"]),
)
hist_exog = historical_exogenous_columns(forecast_train_df)

task.get_logger().report_text(
    "Training NeuralForecast model with "
    f"{len(forecast_train_df)} rows, horizon={forecast_config.horizon}, "
    f"input_size={forecast_config.input_size}, hist_exog={hist_exog}."
)

nf, cv_df, metrics = train_forecast_model(
    forecast_train_df,
    forecast_config,
    hist_exog=hist_exog,
)

cv_results_path = output_dir / "train_cv_results.csv"
cv_df.to_csv(cv_results_path, index=False)
task.upload_artifact("train_cv_results", cv_df)

save_forecast_model(nf, model_dir)
archive_model_dir(model_dir, model_archive_path)

output_model = OutputModel(
    task=task,
    name="neuralforecast_deposit_model",
)
output_model.update_weights(weights_filename=str(model_archive_path))
output_model.set_metadata("model_framework", "NeuralForecast")
output_model.set_metadata("forecast_unique_id", str(params["unique_id"]))
output_model.set_metadata(FORECAST_HORIZON_METADATA_KEY, str(forecast_config.horizon))
output_model.set_metadata("feature_dataset_id", feature_dataset_id)
output_model.set_metadata("feature_task_id", feature_task.id)
output_model.set_metadata("train_task_id", task.id)
output_model.set_metadata("model_archive_format", "zip")

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
output_model.set_metadata("raw_dataset_id", raw_dataset_id)

for metric_name, metric_value in metrics.items():
    if isinstance(metric_value, (int, float)):
        task.get_logger().report_single_value(metric_name, float(metric_value))

training_info = {
    "model_id": output_model.id,
    "model_framework": "NeuralForecast",
    "model_archive": os.path.basename(model_archive_path),
    "feature_dataset_id": feature_dataset_id,
    "raw_dataset_id": raw_dataset_id,
    "n_rows": len(forecast_train_df),
    "target_column": TARGET_COLUMN,
    "forecast_columns": list(forecast_train_df.columns),
    "hist_exog": hist_exog,
    "training_config": forecast_config.__dict__,
    "metrics": metrics,
}
model_card = {
    "algorithm": "NeuralForecast NHITS + NBEATSx",
    "forecast_schema": ["unique_id", "ds", "y"],
    "target_column": TARGET_COLUMN,
    "feature_dataset_id": feature_dataset_id,
    "raw_dataset_id": raw_dataset_id,
    "n_rows": len(forecast_train_df),
    "training_config": forecast_config.__dict__,
    "metrics": metrics,
}
train_summary = {
    "model_id": output_model.id,
    "model_name": output_model.name,
    "model_framework": "NeuralForecast",
    "metrics": metrics,
}
train_lineage = {
    "train_task_id": task.id,
    "feature_task_id": params["feature_task_id"],
    "hpo_task_id": params.get("hpo_task_id"),
    "model_id": output_model.id,
    "feature_dataset_id": feature_dataset_id,
    "raw_dataset_id": raw_dataset_id,
}

task.upload_artifact("model_id", output_model.id)
task.upload_artifact("model_archive", model_archive_path)
task.upload_artifact("feature_dataset_id", feature_dataset_id)
task.upload_artifact("raw_dataset_id", raw_dataset_id)
task.upload_artifact("training_info", training_info)
task.upload_artifact("model_card", model_card)
task.upload_artifact("train_summary", train_summary)
task.upload_artifact("train_lineage", train_lineage)
task.upload_artifact("training_metrics", metrics)

task.get_logger().report_text(
    f"""
    Raw Dataset      : {raw_dataset_id}
    Feature Dataset  : {feature_dataset_id}
    Model ID         : {output_model.id}
    Training rows    : {len(forecast_train_df)}
    """
)
task.get_logger().report_text("Forecast training completed.")

print("Forecast training completed.")

task.flush()
task.close()
