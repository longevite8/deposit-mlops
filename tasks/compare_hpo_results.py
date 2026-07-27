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
# Load Results từ 3 HPO Tasks
# =====================================================

task.get_logger().report_text("📍 Loading HPO results from all models...")

hpo_tasks = {
    "lightgbm": Task.get_task(task_id=params["hpo_lightgbm_task_id"]),
    "nbeatsx": Task.get_task(task_id=params["hpo_nbeatsx_task_id"]),
    "nhits": Task.get_task(task_id=params["hpo_nhits_task_id"]),
}

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

        task.get_logger().report_text(
            f"✅ {model_type.upper()}: Score = {best_score:.6f}"
        )

    except Exception as e:
        task.get_logger().report_text(
            f"❌ {model_type.upper()}: Failed to load - {str(e)}"
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

# Find best model
best_model_type = None
best_score = float("inf")

for model_type, result in hpo_results.items():
    score = result.get("best_score", float("inf"))
    task.get_logger().report_text(
        f"{model_type.upper():12} | Score (MAPE): {score:.6f}"
    )

    if score < best_score:
        best_score = score
        best_model_type = model_type

task.get_logger().report_text("=" * 70)
task.get_logger().report_text(
    f"🏆 WINNER: {best_model_type.upper()}\n   Best Score: {best_score:.6f}"
)
task.get_logger().report_text("=" * 70)

# =====================================================
# BUSINESS LOGIC: End
# =====================================================

# =====================================================
# Upload Results
# =====================================================

task.upload_artifact("best_model_type", best_model_type)
task.upload_artifact("best_score", float(best_score))
task.upload_artifact("all_results", hpo_results)

best_model_params = hpo_results[best_model_type].get("best_params", {})
task.upload_artifact("best_params", best_model_params)

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
