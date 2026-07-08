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
    N_SHAP_SAMPLES,
)

# =====================================================
# IMPORTANT: Đảm bảo pipeline được nhận diện đúng cách
# =====================================================

pipe = PipelineController(
    project=PROJECT_PIPELINE,
    name=TRAINING_PIPELINE_NAME,
    version=DEPLOYMENT_VERSION,
)

# Thêm tags để ClearML nhận diện đây là Pipeline
pipe.task.add_tags(["pipeline", "training", "automated"])

pipe.set_default_execution_queue(SERVICES_QUEUE)

# =====================================================
# Extract (KHÔNG cache - data source có thể thay đổi)
# =====================================================

pipe.add_step(
    name="extract",
    base_task_id=TEMPLATE_EXTRACT_ID,
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,  # ✅ KHÔNG cache vì source data thay đổi
)

# =====================================================
# Feature (Cache - output cố định)
# =====================================================

pipe.add_step(
    name="feature",
    parents=["extract"],
    base_task_id=TEMPLATE_FEATURE_ID,
    parameter_override={
        "General/extract_task_id": "${extract.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,  # ✅ Cache vì output cố định từ extract
)

# =====================================================
# Validate (Cache - validation rules cố định)
# =====================================================

pipe.add_step(
    name="validate",
    parents=["feature"],
    base_task_id=TEMPLATE_VALIDATE_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,  # ✅ Cache - rules không đổi
)

# =====================================================
# Drift (KHÔNG cache - so sánh vs historical data)
# =====================================================

pipe.add_step(
    name="drift",
    parents=["validate"],  # ✅ Chỉ phụ thuộc validate
    base_task_id=TEMPLATE_DRIFT_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,  # ✅ KHÔNG cache - so sánh data thay đổi
)

# =====================================================
# HPO (Cache - hyperparameters cố định)
# =====================================================

pipe.add_step(
    name="hpo",
    parents=["drift"],  # ✅ Chỉ phụ thuộc drift
    base_task_id=TEMPLATE_HPO_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,  # ✅ Cache - hyperparameters cố định
)

# =====================================================
# Train (Cache - model cố định từ input)
# =====================================================

pipe.add_step(
    name="train",
    parents=["hpo"],  # ✅ Chỉ phụ thuộc hpo
    base_task_id=TEMPLATE_TRAIN_ID,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
        "General/hpo_task_id": "${hpo.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,  # ✅ Cache - model cố định
)

# =====================================================
# Evaluate (Cache - metrics cố định)
# =====================================================

pipe.add_step(
    name="evaluate",
    parents=["train"],  # ✅ Chỉ phụ thuộc train
    base_task_id=TEMPLATE_EVALUATE_ID,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
        "General/train_task_id": "${train.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,  # ✅ Cache - metrics cố định
    monitor_metrics=[
        ("MAPE", "mape"),
        ("R2", "r2"),
    ],
)

# =====================================================
# Register (Cache - registration logic cố định)
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
    cache_executed_step=True,  # ✅ Cache - logic cố định
)

# =====================================================
# Explainability (Cache - SHAP analysis cố định)
# =====================================================

pipe.add_step(
    name="explain_model",
    parents=["register"],  # ✅ Phụ thuộc register để chắc chắn nó hoàn thành
    base_task_id=TEMPLATE_EXPLAIN_ID,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
        "General/train_task_id": "${train.id}",
        "General/n_samples": N_SHAP_SAMPLES,
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,  # ✅ Cache - SHAP cố định
)

# =====================================================
# Compare Champion (Cache - so sánh logic cố định)
# =====================================================

pipe.add_step(
    name="compare_champion",
    parents=["register"],
    base_task_id=TEMPLATE_COMPARE_CHAMPION_ID,
    parameter_override={
        "General/register_task_id": "${register.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,  # ✅ Cache - logic cố định
)

# =====================================================
# Promote Champion (Cache - promotion logic cố định)
# =====================================================

pipe.add_step(
    name="promote_champion",
    parents=["compare_champion"],
    base_task_id=TEMPLATE_PROMOTE_CHAMPION_ID,
    parameter_override={
        "General/compare_task_id": "${compare_champion.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,  # ✅ Cache - logic cố định
)

# =====================================================
# CÓ LẺ: FIX QUAN TRỌNG - Ensure pipeline is properly registered
# =====================================================

# Flush trước khi start
pipe.task.flush()

# Start pipeline
pipe.start()

# Log với thêm thông tin
print("✅ Training Pipeline started with caching enabled")
print(f"   Task ID: {pipe.task.id}")
print(f"   Project: {pipe.task.project}")
print(
    f"   UI URL: http://192.168.140.248:8080/projects/{pipe.task.project_id}/experiments/{pipe.task.id}/output/log"
)
