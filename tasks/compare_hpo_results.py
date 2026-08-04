"""
Compare HPO Results - So sánh kết quả HPO từ 3 models, chọn model tốt nhất.
"""

import math

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

hpo_tasks["lightgbm"] = Task.get_task(task_id=params["hpo_lightgbm_task_id"])
hpo_tasks["nbeatsx"] = Task.get_task(task_id=params["hpo_nbeatsx_task_id"])
hpo_tasks["nhits"] = Task.get_task(task_id=params["hpo_nhits_task_id"])

task.get_logger().report_text(
    f"📍 Comparing {len(hpo_tasks)} model(s): {list(hpo_tasks.keys())}"
)

hpo_results = {}
for model_type, hpo_task in hpo_tasks.items():
    summary_artifact_name = f"hpo_{model_type}_summary"

    try:
        hpo_summary = wait_for_artifact(
            hpo_task,
            summary_artifact_name,
            max_retries=10,
            wait_interval=2.0,
            logger_obj=task,
        )

        raw_score = hpo_summary.get("best_score")
        best_params = hpo_summary.get("best_params")

        if raw_score is None:
            raise ValueError(f"Missing 'best_score' in {summary_artifact_name}")

        best_score = float(raw_score)

        if not math.isfinite(best_score):
            raise ValueError(
                f"Invalid best_score={best_score} in {summary_artifact_name}"
            )

        if not isinstance(best_params, dict) or not best_params:
            raise ValueError(
                f"Missing or empty 'best_params' in {summary_artifact_name}"
            )

        hpo_results[model_type] = {
            "best_score": best_score,
            "best_params": best_params,
            "hpo_task_id": hpo_task.id,
            "model_type": hpo_summary.get("model_type", model_type),
        }

        task.get_logger().report_text(
            f"✅ {model_type.upper()}: Best Score = {best_score:.6f}"
        )

    except ValueError as exc:
        task.get_logger().report_text(
            f"❌ {model_type.upper()}: Invalid HPO summary - {exc!s}"
        )
        hpo_results[model_type] = {
            "best_score": float("inf"),
            "hpo_task_id": hpo_task.id,
            "error": str(exc),
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

    if math.isfinite(float(score)):
        valid_models[model_type] = result
    else:
        failed_models[model_type] = result

# Display all results
for model_type, result in hpo_results.items():
    score = float(result.get("best_score", float("inf")))

    if math.isfinite(score):
        status = "✅ VALID"
        score_text = f"{score:.6f}"
    else:
        status = "❌ FAILED"
        score_text = "INVALID"

    task.get_logger().report_text(
        f"{model_type.upper():12} | Score (MAPE): {score_text:>10} | {status}"
    )

task.get_logger().report_text("=" * 70)

# Find best model
if valid_models:
    best_model_type, best_result = min(
        valid_models.items(),
        key=lambda item: item[1]["best_score"],
    )
    best_score = float(best_result["best_score"])

    task.get_logger().report_text(
        f"🏆 WINNER: {best_model_type.upper()}\n   Best Score: {best_score:.6f}"
    )
else:
    error_msg = "NO VALID MODELS! All HPO tasks failed or returned invalid scores."
    task.get_logger().report_text(f"❌ {error_msg}")
    task.get_logger().report_text("Failed models:")

    for model_type, result in failed_models.items():
        hpo_task_id = result.get("hpo_task_id", "N/A")
        error_detail = result.get("error", "Invalid or unavailable HPO result")
        task.get_logger().report_text(
            f"   - {model_type.upper()}: task_id={hpo_task_id}, error={error_detail}"
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
    "best_score": float(best_score),
    "best_params": valid_models[best_model_type]["best_params"],
    "n_models_compared": len(hpo_tasks),
    "n_valid_models": len(valid_models),
    "n_failed_models": len(failed_models),
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

# =====================================================
# Scalars
# =====================================================

task.get_logger().report_single_value("best_score_overall", float(best_score))
model_hash = sum(ord(char) for char in best_model_type) % 100
task.get_logger().report_single_value(
    "model_selected_hash",
    model_hash,
)

# Report each model's score for comparison
for model_type, result in hpo_results.items():
    score = float(result.get("best_score", float("inf")))

    if math.isfinite(score):
        task.get_logger().report_single_value(
            f"score_{model_type}",
            score,
        )

task.flush()
task.close()
