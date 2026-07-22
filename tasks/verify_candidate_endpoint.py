from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from clearml import Task

from config import (
    CLEARML_CANDIDATE_SERVING_ENDPOINT,
    CLEARML_CANDIDATE_SERVING_ENDPOINT_VERSION,
    CLEARML_SERVING_BASE_URL,
    FORECAST_HORIZON,
    PROJECT_TEMPLATE,
    TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_NAME,
)
from helpers import wait_for_artifact
from scripts.py.serving.verify_clearml_serving import run_check


task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_NAME,
    task_type=Task.TaskTypes.qc,
)

params = task.connect(
    {
        "deploy_candidate_serving_task_id": "",
        "base_url": CLEARML_SERVING_BASE_URL,
        "endpoint_prefix": "",
        "endpoint": CLEARML_CANDIDATE_SERVING_ENDPOINT,
        "version": CLEARML_CANDIDATE_SERVING_ENDPOINT_VERSION,
        "payload_json": "",
        "horizon": FORECAST_HORIZON,
    }
)

if not params["deploy_candidate_serving_task_id"]:
    task.get_logger().report_text("Template creation mode.")
    task.close()
    raise SystemExit(0)

deploy_task = Task.get_task(task_id=params["deploy_candidate_serving_task_id"])
deploy_summary = wait_for_artifact(
    deploy_task,
    "candidate_deploy_summary",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)
deploy_lineage = wait_for_artifact(
    deploy_task,
    "candidate_deploy_lineage",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

if not deploy_summary.get("deployed", False):
    verify_summary = {
        "ok": False,
        "skipped": True,
        "reason": "Candidate serving deployment was skipped or failed upstream.",
    }
else:
    verify_summary = run_check(
        argparse.Namespace(
            base_url=str(params["base_url"]),
            endpoint_prefix=str(params["endpoint_prefix"] or ""),
            endpoint=str(deploy_summary.get("endpoint") or params["endpoint"] or ""),
            version=str(
                deploy_summary.get("endpoint_version") or params["version"] or ""
            ),
            payload_json=str(params["payload_json"] or ""),
            horizon=int(params["horizon"]),
        )
    )

verify_lineage = {
    "verify_candidate_endpoint_task_id": task.id,
    "deploy_candidate_serving_task_id": deploy_task.id,
    "candidate_model_id": deploy_lineage.get("candidate_model_id", ""),
    "service_id": deploy_lineage.get("service_id", ""),
    "endpoint": verify_summary.get("endpoint", deploy_lineage.get("endpoint", "")),
    "endpoint_version": verify_summary.get(
        "endpoint_version",
        deploy_lineage.get("endpoint_version", ""),
    ),
}

task.upload_artifact("candidate_verify_summary", verify_summary)
task.upload_artifact("candidate_verify_lineage", verify_lineage)
task.get_logger().report_single_value(
    "candidate_endpoint_ok", int(bool(verify_summary.get("ok")))
)
task.get_logger().report_text(f"Candidate endpoint verification summary: {verify_summary}")
task.flush()
task.close()
