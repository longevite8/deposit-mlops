"""
Model Selection Pipeline - HPO 3 models song song, so sánh, chọn best.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from clearml.automation import PipelineController

from config import (
    CLEARML_SERVER_URL,
    CPU_QUEUE,
    DEPLOYMENT_VERSION,
    N_SHAP_SAMPLES,
    PROJECT_PIPELINE,
    SERVICES_QUEUE,
    TEMPLATE_COMPARE_CHAMPION_ID,
    TEMPLATE_COMPARE_HPO_ID,
    TEMPLATE_DRIFT_ID,
    TEMPLATE_EVALUATE_ID,
    TEMPLATE_EXPLAIN_ID,
    TEMPLATE_EXTRACT_ID,
    TEMPLATE_FEATURE_ID,
    TEMPLATE_HPO_LIGHTGBM_ID,
    TEMPLATE_HPO_NBEATSX_ID,
    TEMPLATE_HPO_NHITS_ID,
    TEMPLATE_PROMOTE_CHAMPION_ID,
    TEMPLATE_REGISTER_ID,
    TEMPLATE_TRAIN_ID,
    TEMPLATE_VALIDATE_ID,
)

timestamp = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y%m%d_%H%M%S")
pipe = PipelineController(
    project=PROJECT_PIPELINE,
    name="Model Selection Pipeline",
    version=DEPLOYMENT_VERSION,
)

pipe.task.set_tags(["pipeline", "model_selection"])
pipe.task.set_comment(f"Model Selection - Compare 3 Models - {timestamp}")
pipe.set_default_execution_queue(SERVICES_QUEUE)

# =====================================================
# Step 1: Extract
# =====================================================

pipe.add_step(
    name="extract",
    base_task_id=TEMPLATE_EXTRACT_ID,
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,
)

# =====================================================
# Step 2: Feature
# =====================================================

pipe.add_step(
    name="feature",
    parents=["extract"],
    base_task_id=TEMPLATE_FEATURE_ID,
    parameter_override={"General/extract_task_id": "${extract.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)

# =====================================================
# Step 3: Validate
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
# Step 4: Drift
# =====================================================

pipe.add_step(
    name="drift",
    parents=["validate"],
    base_task_id=TEMPLATE_DRIFT_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,
)

# =====================================================
# Step 5: HPO 3 Models - SONG SONG
# =====================================================

pipe.add_step(
    name="hpo_lightgbm",
    parents=["drift"],
    base_task_id=TEMPLATE_HPO_LIGHTGBM_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,
)

pipe.add_step(
    name="hpo_nbeatsx",
    parents=["drift"],
    base_task_id=TEMPLATE_HPO_NBEATSX_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,
)

pipe.add_step(
    name="hpo_nhits",
    parents=["drift"],
    base_task_id=TEMPLATE_HPO_NHITS_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,
)

# =====================================================
# Step 6: Compare HPO Results & Select Best Model
# =====================================================

pipe.add_step(
    name="compare_hpo",
    parents=["hpo_lightgbm", "hpo_nbeatsx", "hpo_nhits"],
    base_task_id=TEMPLATE_COMPARE_HPO_ID,
    parameter_override={
        "General/hpo_lightgbm_task_id": "${hpo_lightgbm.id}",
        "General/hpo_nbeatsx_task_id": "${hpo_nbeatsx.id}",
        "General/hpo_nhits_task_id": "${hpo_nhits.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,
)

# =====================================================
# Step 7: Train Best Model
# =====================================================

pipe.add_step(
    name="train",
    parents=["compare_hpo", "feature"],
    base_task_id=TEMPLATE_TRAIN_ID,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
        "General/compare_hpo_task_id": "${compare_hpo.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)

# =====================================================
# Step 8: Evaluate
# =====================================================

pipe.add_step(
    name="evaluate",
    parents=["train"],
    base_task_id=TEMPLATE_EVALUATE_ID,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
        "General/train_task_id": "${train.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
    monitor_metrics=[
        ("MAPE", "mape"),
        ("R2", "r2"),
    ],
)

# =====================================================
# Step 9: Register
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
    cache_executed_step=True,
)

# =====================================================
# Step 10: Explainability (SHAP Analysis)
# =====================================================

pipe.add_step(
    name="explain_model",
    parents=["register"],
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
# Step 11: Compare Champion
# =====================================================

pipe.add_step(
    name="compare_champion",
    parents=["register"],
    base_task_id=TEMPLATE_COMPARE_CHAMPION_ID,
    parameter_override={
        "General/register_task_id": "${register.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)

# =====================================================
# Step 12: Promote Champion
# =====================================================

pipe.add_step(
    name="promote_champion",
    parents=["compare_champion"],
    base_task_id=TEMPLATE_PROMOTE_CHAMPION_ID,
    parameter_override={
        "General/compare_champion_task_id": "${compare_champion.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)

# =====================================================
# Flush trước khi start
# =====================================================

pipe.task.flush()

print("=" * 70)
print("📌 Starting Model Selection Pipeline...")
print("=" * 70)

pipe.start()

pipeline_id = pipe.task.id

print("=" * 70)
print("✅ Model Selection Pipeline started")
print("=" * 70)
print(f"   Task ID: {pipeline_id}")
print("   Pipeline Name: Model Selection Pipeline")
print(f"   Version: {DEPLOYMENT_VERSION}")
print(f"   Timestamp: {timestamp}")
print(f"   UI URL: {CLEARML_SERVER_URL}/tasks/{pipeline_id}")
print("=" * 70)

# =====================================================
# Final flush để đảm bảo final state được lưu
# =====================================================

pipe.task.flush()
