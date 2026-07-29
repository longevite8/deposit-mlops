"""
Model Selection Pipeline - HPO 3 models song song, so sánh, chọn best.
"""

from datetime import datetime

from clearml.automation import PipelineController

from config import (
    CLEARML_SERVER_URL,
    CPU_QUEUE,
    DEPLOYMENT_VERSION,
    PROJECT_PIPELINE,
    SERVICES_QUEUE,
    TEMPLATE_COMPARE_HPO_ID,
    TEMPLATE_DRIFT_ID,
    TEMPLATE_EVALUATE_ID,
    TEMPLATE_EXTRACT_ID,
    TEMPLATE_FEATURE_ID,
    TEMPLATE_HPO_LIGHTGBM_ID,
    # TEMPLATE_HPO_NBEATSX_ID,
    TEMPLATE_HPO_NHITS_ID,
    TEMPLATE_TRAIN_ID,
    TEMPLATE_VALIDATE_ID,
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

# pipe.add_step(
#     name="hpo_nbeatsx",
#     parents=["drift"],
#     base_task_id=TEMPLATE_HPO_NBEATSX_ID,
#     parameter_override={"General/feature_task_id": "${feature.id}"},
#     execution_queue=CPU_QUEUE,
#     cache_executed_step=False,
# )

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
        # "General/hpo_nbeatsx_task_id": "${hpo_nbeatsx.id}",
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
    cache_executed_step=False,
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
    cache_executed_step=False,  # ← Changed (model type varies)
    monitor_metrics=[
        ("MAPE", "mape"),
        ("R2", "r2"),
    ],
)

print("=" * 70)
print("📌 Starting Model Selection Pipeline...")
print("=" * 70)

pipe.start()

pipeline_id = pipe.task.id

print("=" * 70)
print("✅ Model Selection Pipeline started")
print("=" * 70)
print(f"   Pipeline ID: {pipeline_id}")
print(f"   UI URL: {CLEARML_SERVER_URL}/tasks/{pipeline_id}")
print("=" * 70)

pipe.task.flush()
