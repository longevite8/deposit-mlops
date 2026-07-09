from clearml import Task
from datetime import datetime
import time

from config import (
    PROJECT_TEMPLATE,
    SERVICES_QUEUE,
    TEMPLATE_AUTO_RETRAINING_NAME,
    TRAINING_PIPELINE_ID,
    CLEARML_SERVER_URL,
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
# SỬA: Remove old tags TRƯỚC set_tags() để chắc chắn replace
# =====================================================

# Xoá tất cả tags cũ trước
old_tags = new_pipeline.get_tags()
if old_tags:
    new_pipeline.remove_tags(old_tags)  # ← Xoá tags cũ

# Giờ set tags mới (sạch sẽ)
new_pipeline.set_tags(
    [
        "pipeline",
        "training",
        "automated",  # ← Auto triggered (KHÔNG phải manual)
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

# =====================================================
# SỬA: Multiple flush + longer sleep để ensure server update
# =====================================================

# Flush 1: Ghi tags & comment
new_pipeline.flush()
time.sleep(1)  # ← SỬA: 0.5s → 1s

# Verify tags từ memory
task.get_logger().report_text(
    f"✅ Training pipeline cloned: {new_pipeline.id}\n"
    f"   Tags in memory: {new_pipeline.get_tags()}"
)

# Flush 2: Ensure server fully sync
new_pipeline.flush()
time.sleep(1)  # Chờ server process

# =====================================================
# Verify tags từ server (không phải memory)
# =====================================================

try:
    # Fetch fresh task từ server
    server_pipeline = Task.get_task(task_id=new_pipeline.id)
    server_tags = server_pipeline.get_tags()

    task.get_logger().report_text(
        f"✅ Tags verified from server: {server_tags}\n"
        f"   Expected: ['pipeline', 'training', 'automated', ...]\n"
        f"   Match: {set(server_tags) >= {'pipeline', 'training', 'automated'}}"
    )
except Exception as e:
    task.get_logger().report_text(
        f"⚠️ Could not verify tags from server: {str(e)}", level="warning"
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
# QUAN TRỌNG: Keep task alive + Extended monitoring
# =====================================================

max_wait_time = 60
wait_interval = 2
elapsed_time = 0
status_confirmed = False

task.get_logger().report_text(
    f"⏳ Waiting for training pipeline to start (max {max_wait_time}s)..."
)

while elapsed_time < max_wait_time:
    try:
        # Fetch training pipeline task để confirm status
        training_pipeline = Task.get_task(task_id=new_pipeline.id)
        pipeline_status = training_pipeline.get_status()

        # ✅ Verify tags từ server (fresh fetch)
        current_tags = training_pipeline.get_tags()

        task.get_logger().report_text(
            f"   Status: {pipeline_status} | Tags: {current_tags}"
        )

        # Check tags có đúng không
        expected_tags = {"pipeline", "training", "automated"}
        tags_ok = expected_tags.issubset(set(current_tags))

        if not tags_ok:
            task.get_logger().report_text(
                f"   ⚠️ Tags mismatch! Expected {expected_tags}, got {set(current_tags)}",
                level="warning",
            )

        # Nếu status từ queued → in_progress, nghĩa là training pipeline bắt đầu
        if pipeline_status not in ["created", "queued"]:
            task.get_logger().report_text(
                f"✅ Training pipeline status confirmed: {pipeline_status}\n"
                f"   Final tags: {current_tags}"
            )
            status_confirmed = True
            break

        time.sleep(wait_interval)
        elapsed_time += wait_interval

    except Exception as e:
        task.get_logger().report_text(
            f"⚠️ Could not fetch pipeline status: {str(e)}", level="warning"
        )
        time.sleep(wait_interval)
        elapsed_time += wait_interval

# SỬA: Log warning nếu không confirm trong thời gian cho phép
if not status_confirmed:
    task.get_logger().report_text(
        f"⚠️ Training pipeline status not confirmed within {max_wait_time}s\n"
        f"   Pipeline may be queued or have connectivity issues\n"
        f"   Pipeline ID: {new_pipeline.id}",
        level="warning",
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
    "timestamp": timestamp,
    "status_confirmed": status_confirmed,
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
# Final sync trước close
# =====================================================

task.get_logger().report_text(
    f"✅ Training pipeline launched successfully: {new_pipeline.id}\n"
    f"   UI URL: {CLEARML_SERVER_URL}/tasks/{new_pipeline.id}"
)

# Final flush
new_pipeline.flush()
time.sleep(1)

task.get_logger().report_text(
    "🏁 Task closing to release Agent for scheduled pipeline tasks."
)

task.flush()
task.close()
