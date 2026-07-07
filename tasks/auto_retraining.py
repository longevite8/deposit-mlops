from clearml import Task
from datetime import datetime
from config import (
    PROJECT_TEMPLATE,
    SERVICES_QUEUE,
    TEMPLATE_AUTO_RETRAINING_NAME,
    TRAINING_PIPELINE_ID,
)

# =====================================================
# Task Initialization
# =====================================================

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_AUTO_RETRAINING_NAME,
    task_type=Task.TaskTypes.application,
)

# =====================================================
# Parameters & Guard
# =====================================================

params = task.connect(
    {
        "alert_task_id": "",  # Wired to alerting step in production_pipeline.py
    }
)

# Template mode guard: exit if no alert_task_id provided
if not params["alert_task_id"]:
    task.get_logger().report_text("Template creation mode. Exiting.")
    task.close()
    raise SystemExit(0)

# =====================================================
# Data Retrieval
# =====================================================

alert_task = Task.get_task(task_id=params["alert_task_id"])
alert_summary = alert_task.artifacts["alert_summary"].get()
alert_lineage = alert_task.artifacts["alert_lineage"].get()

# Logic Check: Nếu bước alerting không yêu cầu alert/retrain thì thoát
if not alert_summary.get("alert", False):
    task.get_logger().report_text(
        "ℹ️ No retraining required according to alert_summary."
    )
    task.close()
    raise SystemExit(0)

# =====================================================
# Trigger Training Pipeline
# =====================================================

task.get_logger().report_text(
    f"🚀 Triggering Auto-Retraining. Reason: {alert_summary.get('reason')}"
)

# Clone Training Pipeline template
pipeline_template = Task.get_task(task_id=TRAINING_PIPELINE_ID)
new_pipeline = Task.clone(
    source_task=pipeline_template,
    name=f"Auto Retraining - {datetime.now():%Y%m%d_%H%M%S}",
)

# Set parameters for the triggered pipeline (Audit trail)
new_pipeline.set_parameters(
    {
        "Trigger/alert_task_id": alert_task.id,
        "Trigger/monitoring_task_id": alert_lineage.get("monitoring_task_id", ""),
        "Trigger/auto_retraining_task_id": task.id,
        "Trigger/reason": alert_summary.get("reason", ""),
        "Trigger/model_id": alert_lineage.get("model_id", ""),
        "Trigger/feature_dataset_id": alert_lineage.get("feature_dataset_id", ""),
    }
)

new_pipeline.set_tags(
    [
        "pipeline:training",
        "trigger:auto_retraining",
        f"model:{alert_lineage.get('model_id', 'unknown')}",
    ]
)

# Enqueue to Services Queue (thường dùng cho pipeline controllers)
Task.enqueue(
    new_pipeline,
    queue_name=SERVICES_QUEUE,
)

# =====================================================
# Two-Artifact Lineage Pattern & Summary
# =====================================================

retraining_summary = {
    "triggered": True,
    "launched_pipeline_id": new_pipeline.id,
    "reason": alert_summary.get("reason"),
    "model_id": alert_lineage.get("model_id"),
    "feature_dataset_id": alert_lineage.get("feature_dataset_id"),
    "timestamp": str(datetime.now()),
}

retraining_lineage = {
    "auto_retraining_task_id": task.id,
    "alert_task_id": alert_task.id,
    "monitoring_task_id": alert_lineage.get("monitoring_task_id"),
    "launched_pipeline_id": new_pipeline.id,
    "source_model_id": alert_lineage.get("model_id"),
    "source_feature_dataset_id": alert_lineage.get("feature_dataset_id"),
}

task.upload_artifact("retraining_summary", retraining_summary)
task.upload_artifact("retraining_lineage", retraining_lineage)
task.upload_artifact("launched_pipeline_id", new_pipeline.id)

# =====================================================
# Solution 3: Release Agent Immediately
# =====================================================

task.get_logger().report_text(f"✅ Training pipeline launched: {new_pipeline.id}")
task.get_logger().report_text(
    "🏁 Task closing to release Agent for scheduled pipeline tasks."
)

# Đóng task ngay lập tức để Agent có thể quay lại Queue
# và bốc task 'extract' của pipeline vừa trigger, tránh Deadlock.
task.close()
