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

pipe.set_default_execution_queue(SERVICES_QUEUE)

# =====================================================
# Extract
# =====================================================

pipe.add_step(
    name="extract",
    base_task_id=TEMPLATE_EXTRACT_ID,
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)


# =====================================================
# Feature
# =====================================================

pipe.add_step(
    name="feature",
    parents=["extract"],
    base_task_id=TEMPLATE_FEATURE_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/extract_task_id": "${extract.id}",
    },
    cache_executed_step=True,
)

# =====================================================
# Feature
# =====================================================

pipe.add_step(
    name="drift",
    parents=["feature"],
    base_task_id=TEMPLATE_DRIFT_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
    },
)


# =====================================================
# Inference
# =====================================================

pipe.add_step(
    name="inference",
    parents=[
        "feature",
    ],
    base_task_id=TEMPLATE_INFERENCE_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
    },
)

# =====================================================
# Monitoring
# =====================================================

pipe.add_step(
    name="monitoring",
    parents=[
        "inference",
        "drift",
    ],
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
)

# =====================================================
# Alerting
# =====================================================

pipe.add_step(
    name="alerting",
    parents=[
        "monitoring",
    ],
    base_task_id=TEMPLATE_ALERTING_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/monitoring_task_id": "${monitoring.id}",
    },
)


# =====================================================
# Auto retrain
# =====================================================

pipe.add_step(
    name="auto_retraining",
    parents=[
        "alerting",
    ],
    base_task_id=TEMPLATE_AUTO_RETRAINING_ID,
    execution_queue=CPU_QUEUE,
    parameter_override={
        "General/alert_task_id": "${alerting.id}",
    },
)

pipe.start()
