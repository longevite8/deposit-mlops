"""
Compare HPO Results - So sánh kết quả HPO từ 3 models, chọn model tốt nhất.
"""

from clearml import Task

from helpers import wait_for_artifact

task = Task.init(
    project_name="Deposit-CashFlow/Templates",
    task_name="Compare HPO Results",
    task_type=Task.TaskTypes.qc,
)

# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "hpo_lightgbm_task_id": "",
        "hpo_nbeatsx_task_id": "",
        "hpo_nhits_task_id": "",
    }
)

# =====================================================
# Template creation mode
# =====================================================

if (
    not params["hpo_lightgbm_task_id"]
    or not params["hpo_nbeatsx_task_id"]
    or not params["hpo_nhits_task_id"]
):
    task.get_logger().report_text("Template creation mode.")
    task.close()
    raise SystemExit(0)

# =====================================================
# Load Results từ HPO Tasks
# =====================================================

task.get_logger().report_text("📍 Loading HPO results from available models...")

hpo_tasks = {}

# Always require lightgbm (primary model)
if params["hpo_lightgbm_task_id"]:
    hpo_tasks["lightgbm"] = Task.get_task(task_id=params["hpo_lightgbm_task_id"])

# Optional: nbeatsx & nhits (if provided)
if params.get("hpo_nbeatsx_task_id"):
    hpo_tasks["nbeatsx"] = Task.get_task(task_id=params["hpo_nbeatsx_task_id"])

if params.get("hpo_nhits_task_id"):
    hpo_tasks["nhits"] = Task.get_task(task_id=params["hpo_nhits_task_id"])

task.get_logger().report_text(
    f"📍 Comparing {len(hpo_tasks)} model(s): {list(hpo_tasks.keys())}"
)

hpo_results = {}

for model_type, hpo_task in hpo_tasks.items():
    try:
        best_score = wait_for_artifact(
            hpo_task,
            "best_score",
            max_retries=10,
            wait_interval=2.0,
            logger_obj=task,
        )

        best_params = wait_for_artifact(
            hpo_task,
            "best_params",
            max_retries=10,
            wait_interval=2.0,
            logger_obj=task,
        )

        hpo_results[model_type] = {
            "best_score": float(best_score),
            "best_params": best_params,
            "hpo_task_id": hpo_task.id,
        }

        # Check if score is valid (not Infinity)
        if float(best_score) == float("inf"):
            task.get_logger().report_text(
                f"⚠️  {model_type.upper()}: HPO did not converge - all trials may have failed. Score = Infinity"
            )
        else:
            task.get_logger().report_text(
                f"✅ {model_type.upper()}: Score = {best_score:.6f}"
            )

    except Exception as e:
        task.get_logger().report_text(
            f"❌ {model_type.upper()}: Failed to load artifacts - {e!s}"
        )
        hpo_results[model_type] = {
            "best_score": float("inf"),
            "error": str(e),
        }

# =====================================================
# BUSINESS LOGIC: Begin
# =====================================================

task.get_logger().report_text("\n" + "=" * 70)
task.get_logger().report_text("📊 MODEL COMPARISON RESULTS")
task.get_logger().report_text("=" * 70)

# Separate valid and failed models
valid_models = {}
failed_models = {}

for model_type, result in hpo_results.items():
    score = result.get("best_score", float("inf"))
    if score == float("inf"):
        failed_models[model_type] = result
    else:
        valid_models[model_type] = result

# Display all results
for model_type, result in hpo_results.items():
    score = result.get("best_score", float("inf"))
    status = "❌ FAILED" if score == float("inf") else "✅ VALID"
    task.get_logger().report_text(
        f"{model_type.upper():12} | Score (MAPE): {score} | {status}"
    )

task.get_logger().report_text("=" * 70)

# Find best model
if valid_models:
    best_model_type, best_result = min(
        valid_models.items(), key=lambda x: x[1].get("best_score", float("inf"))
    )
    best_score = best_result.get("best_score", float("inf"))
    task.get_logger().report_text(
        f"🏆 WINNER: {best_model_type.upper()}\n   Best Score: {best_score:.6f}"
    )
else:
    # No valid models - fail the task
    error_msg = "NO VALID MODELS! All models have Infinity score (HPO failed for all)."
    task.get_logger().report_text(f"⚠️  {error_msg}")
    task.get_logger().report_text("   Failed models:")
    for model_type, result in failed_models.items():
        hpo_task_id = result.get("hpo_task_id", "N/A")
        task.get_logger().report_text(f"   - {model_type.upper()}: {hpo_task_id}")

    task.get_logger().report_text(
        "❌ Cannot proceed with model selection. Please fix HPO tasks."
    )
    task.close(status="failed")
    raise SystemExit(1)

task.get_logger().report_text("=" * 70)

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

# =====================================================
# Upload Results
# =====================================================

compare_hpo_summary = {
    "best_model_type": best_model_type,
    "best_score": float(best_score) if best_score != float("inf") else best_score,
    "n_models_compared": len(hpo_tasks),
    "all_results": hpo_results,
}

compare_hpo_lineage = {
    "compare_hpo_task_id": task.id,
    "hpo_lightgbm_task_id": params["hpo_lightgbm_task_id"],
    "hpo_nbeatsx_task_id": params.get("hpo_nbeatsx_task_id"),
    "hpo_nhits_task_id": params.get("hpo_nhits_task_id"),
}

task.upload_artifact("compare_hpo_summary", compare_hpo_summary)
task.upload_artifact("compare_hpo_lineage", compare_hpo_lineage)

# Upload best_params only if model is valid
if best_model_type and best_model_type in valid_models:
    best_model_params = hpo_results[best_model_type].get("best_params", {})
    task.upload_artifact("best_params", best_model_params)
else:
    task.get_logger().report_text(
        "⚠️  No valid best_params to upload (no valid models found)"
    )

# =====================================================
# Scalars
# =====================================================

task.get_logger().report_single_value("best_score_overall", float(best_score))
task.get_logger().report_single_value(
    "model_selected_hash", hash(best_model_type) % 100
)

# Report each model's score for comparison
for model_type, result in hpo_results.items():
    score = result.get("best_score", float("inf"))
    task.get_logger().report_single_value(f"score_{model_type}", float(score))

task.flush()
task.close()
