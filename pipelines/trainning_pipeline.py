from clearml.automation import PipelineController
from datetime import datetime
import os
import time

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
    CLEARML_SERVER_URL,
)

# =====================================================
# IMPORTANT: Phát hiện chế độ chạy (manual vs automated)
# =====================================================

parent_task_id = None
is_automated = False
try:
    from clearml import Task

    current_task = Task.current_task()
    if current_task:
        parent_task_id = current_task.id
        is_automated = True
except Exception:
    pass

# Cách 2: Hoặc check từ environment variable
if not is_automated:
    is_automated = os.getenv("CLEARML_AUTO_RETRAINING", "false").lower() == "true"

# Phát hiện chế độ
run_mode = "automated" if is_automated else "manual"

# =====================================================
# Tạo version semantic cố định + timestamp
# =====================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

pipe = PipelineController(
    project=PROJECT_PIPELINE,
    name=TRAINING_PIPELINE_NAME,
    version=DEPLOYMENT_VERSION,
)

# SỬA: Thêm tags khác biệt dựa trên chế độ chạy
if is_automated:
    # Tags cho lần chạy tự động (triggered by auto-retraining)
    pipe.task.add_tags(
        [
            "pipeline",
            "training",
            "automated",
        ]
    )
    run_description = "Auto-triggered Training Pipeline (by auto-retraining task)"
else:
    # Tags cho lần chạy bằng tay (manual)
    pipe.task.add_tags(
        [
            "pipeline",
            "training",
            "manual",
        ]
    )
    run_description = "Manual Training Pipeline (user-initiated)"

# SỬA: Dùng set_comment() thay vì set_description()
pipe.task.set_comment(
    f"{run_description}\nTimestamp: {timestamp}\nRun Mode: {run_mode.upper()}"
)

pipe.set_default_execution_queue(SERVICES_QUEUE)

# =====================================================
# Extract (KHÔNG cache - data source có thể thay đổi)
# =====================================================

pipe.add_step(
    name="extract",
    base_task_id=TEMPLATE_EXTRACT_ID,
    execution_queue=CPU_QUEUE,
    cache_executed_step=False,
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
    cache_executed_step=True,
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
    cache_executed_step=True,
)

# =====================================================
# Drift (KHÔNG cache - so sánh vs historical data)
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
# HPO (Cache - hyperparameters cố định)
# =====================================================

pipe.add_step(
    name="hpo",
    parents=["drift"],
    base_task_id=TEMPLATE_HPO_ID,
    parameter_override={"General/feature_task_id": "${feature.id}"},
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)

# =====================================================
# Train (Cache - model cố định từ input)
# =====================================================

pipe.add_step(
    name="train",
    parents=["hpo"],
    base_task_id=TEMPLATE_TRAIN_ID,
    parameter_override={
        "General/feature_task_id": "${feature.id}",
        "General/hpo_task_id": "${hpo.id}",
    },
    execution_queue=CPU_QUEUE,
    cache_executed_step=True,
)

# =====================================================
# Evaluate (Cache - metrics cố định)
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
    cache_executed_step=True,
)

# =====================================================
# Explainability (Cache - SHAP analysis cố định)
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
    cache_executed_step=True,
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
    cache_executed_step=True,
)

# =====================================================
# FIX QUAN TRỌNG - Ensure pipeline is properly registered
# =====================================================

# Flush trước khi start
pipe.task.flush()

print("=" * 70)
print(f"📌 Starting Training Pipeline ({run_mode.upper()})...")
print("=" * 70)

# Start pipeline
pipe.start()

pipeline_id = pipe.task.id

print("=" * 70)
print(f"✅ Training Pipeline started ({run_mode.upper()})")
print("=" * 70)
print(f"   Task ID: {pipeline_id}")
print(f"   Project: {pipe.task.project}")
print(f"   Pipeline Name: {TRAINING_PIPELINE_NAME}")
print(f"   Version: {DEPLOYMENT_VERSION}")
print(f"   Timestamp: {timestamp}")
print(f"   Run Mode: {run_mode.upper()}")
print(f"   Tags: {pipe.task.get_tags()}")
print(f"   Comment: {run_description}")
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

        if pipeline_status not in ["created", "queued"]:
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
print("✅ Training Pipeline configuration completed successfully!")
print("=" * 70)
print("   Pipeline will continue running in background")
print(f"   Monitor progress at: {CLEARML_SERVER_URL}/tasks/{pipeline_id}")
print("=" * 70)
