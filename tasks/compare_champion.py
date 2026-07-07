from clearml import (
    Task,
    Model,
)

from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_COMPARE_CHAMPION_NAME,
)

from helpers import wait_for_artifact  # THÊM: Import từ helper

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_COMPARE_CHAMPION_NAME,
    task_type=Task.TaskTypes.application,
)

# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "register_task_id": "",
    }
)

# =====================================================
# Template mode
# =====================================================

if not params["register_task_id"]:
    task.close()

    raise SystemExit(0)

# =====================================================
# Load register result
# =====================================================

register_task = Task.get_task(
    task_id=params["register_task_id"],
)

# SỬA: Dùng wait_for_artifact để chắc chắn artifact sẵn sàng
register_summary = wait_for_artifact(
    register_task,
    "register_summary",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

register_lineage = wait_for_artifact(
    register_task,
    "register_lineage",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

candidate_model_id = register_summary.get("model_id")

compare_lineage = {
    "compare_task_id": task.id,
    "register_task_id": params["register_task_id"],
    "train_task_id": register_lineage["train_task_id"],
    "evaluate_task_id": register_lineage["evaluate_task_id"],
    "hpo_task_id": register_lineage["hpo_task_id"],
    "feature_dataset_id": register_lineage["feature_dataset_id"],
    "candidate_model_id": candidate_model_id,
}

# =====================================================
# Candidate not published
# =====================================================

if not register_summary.get("published", False):
    compare_summary = {
        "candidate_exists": False,
        "candidate_win": False,
        "status": "FAIL",
        "reason": "Candidate not published",
    }
    task.upload_artifact("compare_summary", compare_summary)
    task.upload_artifact(
        "compare_lineage", compare_lineage
    )  # Fix KeyError cho task sau
    task.close()
    raise SystemExit(0)

# =====================================================
# Candidate metrics
# =====================================================

candidate_mape = register_summary["mape"]
candidate_r2 = register_summary["r2"]

# =====================================================
# Current champion
# =====================================================

champion_models = Model.query_models(
    tags=["champion"],
    only_published=True,
    max_results=1,
)

# =====================================================
# First champion
# =====================================================

if len(champion_models) == 0:
    compare_summary = {
        "candidate_exists": True,
        "candidate_win": True,
        "status": "FIRST_CHAMPION",
        "reason": "No champion",
        "candidate_model_id": candidate_model_id,
    }

    task.upload_artifact("compare_summary", compare_summary)
    task.upload_artifact("compare_lineage", compare_lineage)
    task.get_logger().report_text("✅ First champion promoted.")
    task.close()
    raise SystemExit(0)

# =====================================================
# Champion metrics
# =====================================================

champion_model = champion_models[0]
champion_model_id = champion_model.id
champion_metadata = champion_model.get_all_metadata()
champion_mape = float(champion_metadata["mape"]["value"])
champion_r2 = float(champion_metadata["r2"]["value"])

# =====================================================
# Compare logic
# =====================================================

candidate_win = candidate_mape < champion_mape and candidate_r2 > champion_r2

# =====================================================
# Upload artifacts
# =====================================================

# 1. Summary Artifact
compare_summary = {
    "candidate_win": candidate_win,
    "candidate_model_id": candidate_model_id,
    "champion_model_id": champion_model_id,
    "comparison_metric": "mape",
    "status": "SUCCESS",
}
task.upload_artifact("compare_summary", compare_summary)

# 2. Lineage Artifact
compare_lineage["champion_model_id"] = champion_model_id
task.upload_artifact("compare_lineage", compare_lineage)

task.get_logger().report_text(f"✅ Compare Champion completed. Win: {candidate_win}")

# THÊM: Đồng bộ hoàn toàn trước khi kết thúc
task.flush()

task.close()
