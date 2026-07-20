from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clearml import Task

from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_ALERTING_NAME,
)

from helpers import wait_for_artifact
from business.alerting import (
    determine_alert_status,
    build_alert_subject,
    format_email_body,
    send_email,  # THÊM: Import trực tiếp từ business layer
)
from datetime import datetime


# =====================================================
# Task init
# =====================================================

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_ALERTING_NAME,
    task_type=Task.TaskTypes.qc,
)

# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "monitoring_task_id": "",
    }
)

# =====================================================
# Template creation mode
# =====================================================

if not params["monitoring_task_id"]:
    task.get_logger().report_text("Template creation mode.")

    task.close()

    raise SystemExit(0)


# =====================================================
# Load Data & Artifacts
# =====================================================

monitoring_task = Task.get_task(task_id=params["monitoring_task_id"])
monitoring_summary = wait_for_artifact(monitoring_task, "monitoring_summary")
monitoring_lineage = wait_for_artifact(monitoring_task, "monitoring_lineage")

drift_task = Task.get_task(task_id=monitoring_lineage["drift_task_id"])
drift_summary = wait_for_artifact(drift_task, "drift_summary")

monitoring_metrics = wait_for_artifact(
    monitoring_task, "monitoring_metrics"
)  # Thêm load metrics

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

alert_needed, reason = determine_alert_status(monitoring_summary, drift_summary)
subject = build_alert_subject(alert_needed)
email_body = format_email_body(
    monitoring_summary, monitoring_lineage, monitoring_metrics, alert_needed, reason
)
email_sent = send_email(subject, email_body, logger=task.get_logger())

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

# =====================================================
# ARTIFACTS & LOGS
# =====================================================
alert_summary = {
    "alert": alert_needed,
    "status": monitoring_summary["status"],
    "severity": "HIGH" if alert_needed else "INFO",
    "need_retraining": alert_needed,
    "reason": reason,
    "email_sent": email_sent,
    "timestamp": datetime.utcnow().isoformat(),
}

alert_lineage = {
    "alert_task_id": task.id,
    "monitoring_task_id": params["monitoring_task_id"],
    "model_id": monitoring_lineage.get("model_id"),
    "feature_dataset_id": monitoring_lineage.get("feature_dataset_id"),
}

task.upload_artifact("alert_summary", alert_summary)
task.upload_artifact("alert_lineage", alert_lineage)

markdown = f"""
# Alert Dashboard
## Alert Status
| Item | Value |
|------|-------|
| Alert | **{alert_summary["alert"]}** |
| Severity | **{alert_summary["severity"]}** |
| Email Sent | {alert_summary["email_sent"]} |
## Model
| Item | Value |
|------|-------|
| Model ID | {alert_lineage["model_id"]} |
## Reason
{alert_summary["reason"]}
"""
task.get_logger().report_text(markdown)

task.flush()
task.close()
