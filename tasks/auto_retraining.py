from clearml import Task
from datetime import datetime
import time

from config import (
    PROJECT_TEMPLATE,
    SERVICES_QUEUE,
    TEMPLATE_AUTO_RETRAINING_NAME,
    TRAINING_PIPELINE_ID,
    CLEARML_SERVER_URL,  # ← THÊM IMPORT
)

from helpers import wait_for_artifact

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

# SỬA: Dùng wait_for_artifact để chắc chắn artifact sẵn sàng
alert_summary = wait_for_artifact(
    alert_task,
    "alert_summary",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

alert_lineage = wait_for_artifact(
    alert_task,
    "alert_lineage",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Clone Training Pipeline template
pipeline_template = Task.get_task(task_id=TRAINING_PIPELINE_ID)
new_pipeline = Task.clone(
    source_task=pipeline_template,
    name=f"Auto Retraining - {timestamp}",
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

# =====================================================
# SỬA: Dùng set_tags() để REPLACE tags (không phải add_tags)
# =====================================================

new_pipeline.set_tags(
    [
        "pipeline",
        "training",
        "automated",  # ← Auto triggered (không phải manual)
        f"run_{timestamp}",
        "triggered_by_alerting",
    ]
)

new_pipeline.set_comment(
    f"Automated Training Pipeline (by Auto Retraining)\n"
    f"Timestamp: {timestamp}\n"
    f"Run Mode: AUTOMATED\n"
    f"Reason: {alert_summary.get('reason', 'Scheduled retraining')}"
)

# Flush để đảm bảo metadata được lưu
new_pipeline.flush()

task.get_logger().report_text(
    f"✅ Training pipeline cloned: {new_pipeline.id}\n"
    f"   Tags: {new_pipeline.get_tags()}"
)

# =====================================================
# Enqueue to Services Queue
# =====================================================

Task.enqueue(
    new_pipeline,
    queue_name=SERVICES_QUEUE,
)

task.get_logger().report_text(
    f"📤 Training pipeline enqueued to '{SERVICES_QUEUE}' queue"
)

# =====================================================
# QUAN TRỌNG: Keep task alive để training pipeline được confirm
# =====================================================

# Chờ training pipeline được enqueue đúng cách (tối đa 30 giây)
max_wait_time = 30
wait_interval = 2
elapsed_time = 0

task.get_logger().report_text(
    f"⏳ Waiting for training pipeline to start (max {max_wait_time}s)..."
)

while elapsed_time < max_wait_time:
    try:
        # Fetch training pipeline task để confirm status
        training_pipeline = Task.get_task(task_id=new_pipeline.id)
        pipeline_status = training_pipeline.get_status()

        task.get_logger().report_text(f"   Training pipeline status: {pipeline_status}")

        # Nếu status từ queued → in_progress, nghĩa là training pipeline bắt đầu
        if pipeline_status not in ["created", "queued"]:
            task.get_logger().report_text(
                f"✅ Training pipeline status confirmed: {pipeline_status}"
            )
            break

        time.sleep(wait_interval)
        elapsed_time += wait_interval

    except Exception as e:
        task.get_logger().report_text(
            f"⚠️ Could not fetch pipeline status: {str(e)}", level="warning"
        )
        time.sleep(wait_interval)
        elapsed_time += wait_interval

# =====================================================
# Two-Artifact Lineage Pattern & Summary
# =====================================================

retraining_summary = {
    "triggered": True,
    "launched_pipeline_id": new_pipeline.id,
    "reason": alert_summary.get("reason"),
    "model_id": alert_lineage.get("model_id"),
    "feature_dataset_id": alert_lineage.get("feature_dataset_id"),
    "timestamp": timestamp,
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

task.get_logger().report_text(
    f"✅ Training pipeline launched successfully: {new_pipeline.id}\n"
    f"   UI URL: {CLEARML_SERVER_URL}/tasks/{new_pipeline.id}"
)
task.get_logger().report_text(
    "🏁 Task closing to release Agent for scheduled pipeline tasks."
)

# THÊM: Đồng bộ hoàn toàn trước khi kết thúc
task.flush()

task.close()
