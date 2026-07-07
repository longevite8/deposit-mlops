from clearml.automation import PipelineController

from config import (
    PROJECT_PIPELINE,
    SERVICES_QUEUE,
    CPU_QUEUE,
    TEMPLATE_EXTRACT_ID,
    TEMPLATE_FEATURE_ID,
    TEMPLATE_VALIDATE_ID,
    TEMPLATE_DRIFT_ID,
    TEMPLATE_HPO_ID,
    TEMPLATE_TRAIN_ID,
    TEMPLATE_EVALUATE_ID,
    TEMPLATE_REGISTER_ID,
    TEMPLATE_COMPARE_CHAMPION_ID,
    TEMPLATE_PROMOTE_CHAMPION_ID,
    TEMPLATE_EXPLAIN_ID,
    TRAINING_PIPELINE_NAME,
    DEPLOYMENT_VERSION,
    N_SHAP_SAMPLES,  # ← THÊM DÒNG NÀY
)


pipe = PipelineController(
    project=PROJECT_PIPELINE,
    name=TRAINING_PIPELINE_NAME,
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
    parameter_override={
        "General/extract_task_id": "${extract.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)


# =====================================================
# Validate
# =====================================================

pipe.add_step(
    name="validate",
    parents=["feature"],
    base_task_id=TEMPLATE_VALIDATE_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)


# =====================================================
# Drift
# =====================================================

pipe.add_step(
    name="drift",
    parents=["validate", "feature"],  # ✅ Thêm "feature" vì dùng ${feature.id}
    base_task_id=TEMPLATE_DRIFT_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)


# =====================================================
# HPO
# =====================================================

pipe.add_step(
    name="hpo",
    parents=["drift", "feature"],  # ✅ Thêm "feature"
    base_task_id=TEMPLATE_HPO_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)

# =====================================================
# Train
# =====================================================

pipe.add_step(
    name="train",
    parents=["hpo", "feature"],  # ✅ Thêm "feature"
    base_task_id=TEMPLATE_TRAIN_ID,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
        "General/hpo_task_id": "${hpo.id}",
    },
    execution_queue=CPU_QUEUE,
)

# =====================================================
# Evaluate
# =====================================================

pipe.add_step(
    name="evaluate",
    parents=["train", "feature"],  # ✅ Thêm "feature"
    base_task_id=TEMPLATE_EVALUATE_ID,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
        "General/train_task_id": "${train.id}",
    },
    execution_queue=CPU_QUEUE,
    monitor_metrics=[
        ("MAPE", "mape"),
        ("R2", "r2"),
    ],
)

# =====================================================
# Register
# =====================================================

pipe.add_step(
    name="register",
    parents=["evaluate"],
    base_task_id=TEMPLATE_REGISTER_ID,
    parameter_override={
        "General/train_task_id": "${train.id}",
        "General/evaluate_task_id": "${evaluate.id}",
    },
    execution_queue=CPU_QUEUE,
)

# =====================================================
# Explainability
# =====================================================

pipe.add_step(
    name="explain_model",
    parents=["train", "feature"],  # ✅ Thêm "feature"
    base_task_id=TEMPLATE_EXPLAIN_ID,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
        "General/train_task_id": "${train.id}",
        "General/n_samples": N_SHAP_SAMPLES,
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)

# =====================================================
# Compare Champion
# =====================================================

pipe.add_step(
    name="compare_champion",
    parents=["register"],
    base_task_id=TEMPLATE_COMPARE_CHAMPION_ID,
    parameter_override={
        "General/register_task_id": "${register.id}",
    },
    execution_queue=CPU_QUEUE,
)

# =====================================================
# Promote Champion
# =====================================================

pipe.add_step(
    name="promote_champion",
    parents=["compare_champion"],
    base_task_id=TEMPLATE_PROMOTE_CHAMPION_ID,
    parameter_override={
        "General/compare_task_id": "${compare_champion.id}",
    },
    execution_queue=CPU_QUEUE,
)

pipe.start()
