from clearml import PipelineController

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
)


pipe = PipelineController(
    project=PROJECT_PIPELINE,
    name=PRODUCTION_PIPELINE_NAME,
    version=DEPLOYMENT_VERSION,
)

# Thêm tags để ClearML nhận diện đây là Pipeline
pipe.task.add_tags(["pipeline", "production", "automated"])

pipe.set_default_execution_queue(SERVICES_QUEUE)

# =====================================================
# Extract (KHÔNG cache - data mới mỗi ngày)
# =====================================================

pipe.add_step(
    name="extract",
    base_task_id=TEMPLATE_EXTRACT_ID,
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,  # ✅ KHÔNG cache - data thay đổi mỗi ngày
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
    cache_executed_step=False,  # ✅ KHÔNG cache - phụ thuộc extract mới
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
    cache_executed_step=False,  # ✅ KHÔNG cache - so sánh thay đổi
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
        "General/feature_task_id": "${feature.id}",  # ✅ ĐÃ CÓ
    },
    cache_executed_step=False,  # ✅ KHÔNG cache - predictions mới
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
    cache_executed_step=False,  # ✅ KHÔNG cache - metrics mới
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
    cache_executed_step=False,  # ✅ KHÔNG cache - alert thay đổi
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
    cache_executed_step=False,  # ✅ KHÔNG cache - retraining decision thay đổi
)

# =====================================================
# FIX: Ensure pipeline is properly registered
# =====================================================

# Flush trước khi start
pipe.task.flush()

# Start pipeline
pipe.start()

# Log với thêm thông tin
print("✅ Production Pipeline started")
print(f"   Task ID: {pipe.task.id}")
print(f"   Project: {pipe.task.project}")
# SỬA: Dùng get_project_id() hoặc bỏ project_id
try:
    project_id = pipe.task.get_project_id()
    print(
        f"   UI URL: http://192.168.140.248:8080/projects/{project_id}/experiments/{pipe.task.id}/output/log"
    )
except Exception:
    # Nếu get_project_id() fail, dùng project name
    print(f"   UI URL: http://192.168.140.248:8080/tasks/{pipe.task.id}")
