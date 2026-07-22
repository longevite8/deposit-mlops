from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from clearml import Task

from config import (
    AUTO_CREATE_CLEARML_SERVING_SERVICE,
    CLEARML_BASE_SERVING_URL,
    CLEARML_CANDIDATE_SERVING_ALIAS,
    CLEARML_CANDIDATE_SERVING_ENDPOINT,
    CLEARML_CANDIDATE_SERVING_ENDPOINT_VERSION,
    CLEARML_SERVING_BASE_URL,
    CLEARML_SERVING_ENGINE,
    CLEARML_SERVING_METRIC_LOG_FREQ,
    CLEARML_SERVING_PREPROCESS,
    CLEARML_SERVING_PROJECT,
    CLEARML_SERVING_SERVICE_ID,
    CLEARML_SERVING_SERVICE_NAME,
    FORECAST_HORIZON,
    PROJECT_TEMPLATE,
    TEMPLATE_DEPLOY_CANDIDATE_SERVING_NAME,
)
from helpers import wait_for_artifact
from scripts.py.serving.deploy_clearml_serving import deploy_model


task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_DEPLOY_CANDIDATE_SERVING_NAME,
    task_type=Task.TaskTypes.service,
)

params = task.connect(
    {
        "register_task_id": "",
        "service_id": CLEARML_SERVING_SERVICE_ID,
        "auto_create_service": AUTO_CREATE_CLEARML_SERVING_SERVICE,
        "service_name": CLEARML_SERVING_SERVICE_NAME,
        "service_project": CLEARML_SERVING_PROJECT,
        "base_serving_url": CLEARML_BASE_SERVING_URL,
        "inference_base_url": CLEARML_SERVING_BASE_URL,
        "metric_log_freq": CLEARML_SERVING_METRIC_LOG_FREQ,
        "endpoint_prefix": "",
        "endpoint": CLEARML_CANDIDATE_SERVING_ENDPOINT,
        "endpoint_version": CLEARML_CANDIDATE_SERVING_ENDPOINT_VERSION,
        "preprocess": CLEARML_SERVING_PREPROCESS,
        "engine": CLEARML_SERVING_ENGINE,
        "alias": CLEARML_CANDIDATE_SERVING_ALIAS,
        "horizon": FORECAST_HORIZON,
    }
)


def truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").lower() in {"1", "true", "yes", "y"}


if not params["register_task_id"]:
    task.get_logger().report_text("Template creation mode.")
    task.close()
    raise SystemExit(0)

register_task = Task.get_task(task_id=params["register_task_id"])
register_summary = wait_for_artifact(
    register_task,
    "register_summary",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)
register_lineage = wait_for_artifact(
    register_task,
    "register_lineage",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

if not register_summary.get("published", False):
    deploy_summary = {
        "deployed": False,
        "skipped": True,
        "reason": "Candidate model was not published.",
        "register_task_id": register_task.id,
        "candidate_model_id": register_lineage.get("model_id", ""),
    }
    deploy_lineage = {
        "deploy_candidate_serving_task_id": task.id,
        "register_task_id": register_task.id,
        "candidate_model_id": register_lineage.get("model_id", ""),
        "feature_dataset_id": register_lineage.get("feature_dataset_id", ""),
    }
    task.upload_artifact("candidate_deploy_summary", deploy_summary)
    task.upload_artifact("candidate_deploy_lineage", deploy_lineage)
    task.get_logger().report_text(deploy_summary["reason"])
    task.close()
    raise SystemExit(0)

model_id = register_summary["model_id"]
deployment = deploy_model(
    argparse.Namespace(
        model_id=model_id,
        alias=str(params["alias"]),
        service_id=str(params["service_id"] or ""),
        auto_create_service=truthy(params["auto_create_service"]),
        service_name=str(params["service_name"]),
        service_project=str(params["service_project"]),
        base_serving_url=str(params["base_serving_url"] or ""),
        inference_base_url=str(params["inference_base_url"]),
        metric_log_freq=float(params["metric_log_freq"]),
        endpoint_prefix=str(params["endpoint_prefix"] or ""),
        endpoint=str(params["endpoint"] or ""),
        endpoint_version=str(params["endpoint_version"] or model_id),
        preprocess=str(params["preprocess"]),
        engine=str(params["engine"]),
        horizon=int(params["horizon"]),
    )
)

deploy_summary = {
    **deployment,
    "register_task_id": register_task.id,
    "candidate_model_id": model_id,
}
deploy_lineage = {
    "deploy_candidate_serving_task_id": task.id,
    "register_task_id": register_task.id,
    "candidate_model_id": model_id,
    "feature_dataset_id": register_lineage.get("feature_dataset_id", ""),
    "service_id": deployment["service_id"],
    "endpoint": deployment["endpoint"],
    "endpoint_version": deployment["endpoint_version"],
}

task.upload_artifact("candidate_deploy_summary", deploy_summary)
task.upload_artifact("candidate_deploy_lineage", deploy_lineage)
task.get_logger().report_text(
    f"Deployed candidate model {model_id} to {deployment['endpoint_url']}"
)
task.flush()
task.close()
