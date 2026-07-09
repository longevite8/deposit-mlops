from clearml import PipelineController
from datetime import datetime
import time

from config import (
    PROJECT_PIPELINE,
    CPU_QUEUE,
    SERVICES_QUEUE,
    TEMPLATE_EXTRACT_ID,
    TEMPLATE_FEATURE_ID,
    TEMPLATE_INFERENCE_ID,
    TEMPLATE_DRIFT_ID,
    TEMPLATE_MONITORING_ID,
    TEMPLATE_ALERTING_ID,
    TEMPLATE_AUTO_RETRAINING_ID,
    DEPLOYMENT_VERSION,
    PRODUCTION_PIPELINE_NAME,
    CLEARML_SERVER_URL,
)

# =====================================================
# IMPORTANT: Production Pipeline giữ version cố định
# =====================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

pipe = PipelineController(
    project=PROJECT_PIPELINE,
    # name=PRODUCTION_PIPELINE_NAME,
    name=f"{PRODUCTION_PIPELINE_NAME} - {datetime.now():%Y%m%d_%H%M%S}",
    version=DEPLOYMENT_VERSION,
)


# Thêm tags để ClearML nhận diện đây là Pipeline
pipe.task.add_tags(
    [
        "pipeline",
        "production",
    ]
)

# SỬA: Dùng set_comment() thay vì set_description()
pipe.task.set_comment(f"Production Pipeline (automated)\nTimestamp: {timestamp}")

pipe.set_default_execution_queue(SERVICES_QUEUE)

# =====================================================
# Extract (KHÔNG cache - data mới mỗi ngày)
# =====================================================

pipe.add_step(
    name="extract",
    base_task_id=TEMPLATE_EXTRACT_ID,
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,
)

# =====================================================
# Feature (Cache - nhưng tuỳ ngữ cảnh)
# =====================================================

pipe.add_step(
    name="feature",
    parents=["extract"],
    base_task_id=TEMPLATE_FEATURE_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/extract_task_id": "${extract.id}",
    },
    cache_executed_step=False,
)

# =====================================================
# Drift Detection (KHÔNG cache - so sánh mới mỗi lần)
# =====================================================

pipe.add_step(
    name="drift",
    parents=["feature"],
    base_task_id=TEMPLATE_DRIFT_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
    },
    cache_executed_step=False,
)

# =====================================================
# Inference (KHÔNG cache - predictions mới mỗi lần)
# =====================================================

pipe.add_step(
    name="inference",
    parents=["feature"],
    base_task_id=TEMPLATE_INFERENCE_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
    },
    cache_executed_step=False,
)

# =====================================================
# Monitoring (KHÔNG cache - metrics monitoring thay đổi)
# =====================================================

pipe.add_step(
    name="monitoring",
    parents=["inference", "drift"],
    base_task_id=TEMPLATE_MONITORING_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
        "General/inference_task_id": "${inference.id}",
        "General/drift_task_id": "${drift.id}",
    },
    monitor_metrics=[
        ("MAPE", "MAPE"),
        ("R2", "R2"),
        ("drift_ratio", "drift_ratio"),
        ("need_retraining", "need_retraining"),
    ],
    cache_executed_step=False,
)

# =====================================================
# Alerting (KHÔNG cache - quyết định alert mới mỗi lần)
# =====================================================

pipe.add_step(
    name="alerting",
    parents=["monitoring"],
    base_task_id=TEMPLATE_ALERTING_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/monitoring_task_id": "${monitoring.id}",
    },
    cache_executed_step=False,
)

# =====================================================
# Auto Retraining (KHÔNG cache - quyết định retraining mới)
# =====================================================

pipe.add_step(
    name="auto_retraining",
    parents=["alerting"],
    base_task_id=TEMPLATE_AUTO_RETRAINING_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/alert_task_id": "${alerting.id}",
    },
    cache_executed_step=False,
)

# =====================================================
# FIX QUAN TRỌNG - Ensure pipeline is properly registered
# =====================================================

# Flush trước khi start
pipe.task.flush()

print("=" * 70)
print("📌 Starting Production Pipeline...")
print("=" * 70)

# Start pipeline
pipe.start()

pipeline_id = pipe.task.id

print("=" * 70)
print("✅ Production Pipeline started")
print("=" * 70)
print(f"   Task ID: {pipeline_id}")
print(f"   Project: {pipe.task.project}")
print(f"   Pipeline Name: {PRODUCTION_PIPELINE_NAME}")
print(f"   Version: {DEPLOYMENT_VERSION}")
print(f"   Timestamp: {timestamp}")
print(f"   UI URL: {CLEARML_SERVER_URL}/tasks/{pipeline_id}")
print("=" * 70)

# =====================================================
# QUAN TRỌNG: Keep task alive để ClearML UI update status
# =====================================================

# Chờ pipeline bắt đầu chạy (tối đa 30 giây)
max_wait_time = 30
wait_interval = 2
elapsed_time = 0

print(f"\n⏳ Waiting for pipeline to start (max {max_wait_time}s)...")

while elapsed_time < max_wait_time:
    try:
        # Lấy status của pipeline
        from clearml import Task

        current_pipeline_task = Task.get_task(task_id=pipeline_id)
        pipeline_status = current_pipeline_task.get_status()

        print(f"   Pipeline status: {pipeline_status}")

        if pipeline_status not in ["created", "queued", "in_progress"]:
            print(f"✅ Pipeline status confirmed: {pipeline_status}")
            break

        time.sleep(wait_interval)
        elapsed_time += wait_interval

    except Exception as e:
        print(f"⚠️ Could not fetch pipeline status: {str(e)}")
        time.sleep(wait_interval)
        elapsed_time += wait_interval

# =====================================================
# Final flush để đảm bảo task được lưu trữ đầy đủ
# =====================================================

print("\n📤 Finalizing task...")
pipe.task.flush()
pipe.task.close()

print("=" * 70)
print("✅ Production Pipeline configuration completed successfully!")
print("=" * 70)
print("   Pipeline will continue running in background")
print(f"   Monitor progress at: {CLEARML_SERVER_URL}/tasks/{pipeline_id}")
print("=" * 70)
