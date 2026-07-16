from pathlib import Path

import pandas as pd
from clearml import Dataset, InputModel, Task

from business.evaluate import check_quality_gate
from business.forecasting import (
    add_calendar_features,
    calculate_forecast_metrics,
    historical_exogenous_columns,
    load_forecast_model_from_archive,
    prepare_forecast_frame,
)
from config import (
    FORECAST_EVAL_HORIZON,
    FORECAST_UNIQUE_ID,
    MAPE_THRESHOLD,
    PROJECT_TEMPLATE,
    R2_THRESHOLD,
    TEMPLATE_EVALUATE_NAME,
)
from helpers import wait_for_artifact


task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_EVALUATE_NAME,
    task_type=Task.TaskTypes.qc,
)


params = task.connect(
    {
        "feature_task_id": "",
        "train_task_id": "",
        "mape_threshold": MAPE_THRESHOLD,
        "r2_threshold": R2_THRESHOLD,
        "eval_horizon": FORECAST_EVAL_HORIZON,
        "unique_id": FORECAST_UNIQUE_ID,
    }
)


if not params["feature_task_id"] or not params["train_task_id"]:
    task.get_logger().report_text("Template creation mode.")
    task.close()
    raise SystemExit(0)


mape_threshold = float(params["mape_threshold"])
r2_threshold = float(params["r2_threshold"])
eval_horizon = int(params["eval_horizon"])


feature_task = Task.get_task(task_id=params["feature_task_id"])
feature_dataset_id = wait_for_artifact(
    feature_task,
    "feature_dataset_id",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)
feature_dataset = Dataset.get(dataset_id=feature_dataset_id)
local_path = Path(feature_dataset.get_local_copy())

train_df = pd.read_parquet(local_path / "train.parquet")
valid_df = pd.read_parquet(local_path / "valid.parquet")
test_df = pd.read_parquet(local_path / "test.parquet")

history_df = prepare_forecast_frame(
    pd.concat([train_df, valid_df], ignore_index=True),
    unique_id=str(params["unique_id"]),
)
test_forecast_df = prepare_forecast_frame(
    test_df,
    unique_id=str(params["unique_id"]),
)

train_task = Task.get_task(task_id=params["train_task_id"])
model_id = train_task.artifacts["model_id"].get()
input_model = InputModel(model_id=model_id)
model_archive_path = input_model.get_local_copy()
model = load_forecast_model_from_archive(model_archive_path)

forecasts = model.predict(df=history_df)
forecasts = add_calendar_features(forecasts)
for column in historical_exogenous_columns(history_df):
    if column not in forecasts.columns:
        forecasts[column] = 0
forecasts = (
    forecasts.sort_values(["unique_id", "ds"])
    .groupby("unique_id")
    .head(eval_horizon)
    .reset_index(drop=True)
)
forecast_with_actuals = forecasts.merge(
    test_forecast_df[["unique_id", "ds", "y"]],
    on=["unique_id", "ds"],
    how="inner",
)

metrics = calculate_forecast_metrics(
    forecast_with_actuals,
    eval_horizon=eval_horizon,
)
if not metrics:
    metrics = {
        "mape": float("inf"),
        "mae": float("inf"),
        "rmse": float("inf"),
        "r2": float("-inf"),
    }
passed = check_quality_gate(metrics, mape_threshold, r2_threshold)

task.upload_artifact("forecasts", forecasts)
task.upload_artifact("forecast_with_actuals", forecast_with_actuals)

evaluate_summary = {
    "feature_dataset_id": feature_dataset_id,
    "model_id": model_id,
    "model_framework": "NeuralForecast",
    "passed": passed,
    "mape_threshold": mape_threshold,
    "r2_threshold": r2_threshold,
    "eval_horizon": eval_horizon,
    "n_forecast_rows": len(forecasts),
    "n_evaluated_rows": len(forecast_with_actuals),
    **metrics,
}
evaluate_lineage = {
    "evaluate_task_id": task.id,
    "train_task_id": params["train_task_id"],
    "feature_task_id": params["feature_task_id"],
    "model_id": model_id,
    "feature_dataset_id": feature_dataset_id,
}

task.upload_artifact("evaluate_summary", evaluate_summary)
task.upload_artifact("evaluate_lineage", evaluate_lineage)

for metric_name, metric_value in metrics.items():
    if isinstance(metric_value, (int, float)):
        task.get_logger().report_single_value(metric_name, float(metric_value))

task.get_logger().report_single_value("quality_gate_passed", int(passed))
task.get_logger().report_text(f"Evaluation completed. Passed: {passed}")

task.flush()
task.close()
