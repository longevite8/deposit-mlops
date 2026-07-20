from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clearml.backend_api.session.client import APIClient
from datetime import datetime
from clearml import (
    Task,
    Model,
)

from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_PROMOTE_CHAMPION_NAME,
)

from helpers import wait_for_artifact  # THÊM: Import từ helper

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_PROMOTE_CHAMPION_NAME,
    task_type=Task.TaskTypes.application,
)

# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "compare_task_id": "",
    }
)
# =====================================================
# Template mode
# =====================================================

if not params["compare_task_id"]:
    task.close()

    raise SystemExit(0)

# =====================================================
# Load register result
# =====================================================

compare_task = Task.get_task(task_id=params["compare_task_id"])

# Dùng wait_for_artifact để chắc chắn artifact sẵn sàng
compare_summary = wait_for_artifact(
    compare_task,
    "compare_summary",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

compare_lineage = wait_for_artifact(
    compare_task,
    "compare_lineage",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

candidate_win = compare_summary.get("candidate_win", False)

if not candidate_win:
    task.get_logger().report_text(
        "Candidate model did not win or does not exist. Skipping promotion."
    )
    task.close()
    raise SystemExit(0)

compare_lineage = compare_task.artifacts["compare_lineage"].get()
candidate_win = compare_summary["candidate_win"]

# =====================================================
# Nothing to promote
# =====================================================

if not candidate_win:
    task.get_logger().report_text("No new candidate model.")
    task.close()

    raise SystemExit(0)

# =====================================================
# Old champion
# =====================================================

old_models = Model.query_models(
    tags=["champion"],
    only_published=True,
    max_results=1,
)

previous_champion_model_id = None
previous_promote_task_id = None

client = APIClient()
new_model_id = compare_summary["candidate_model_id"]

if old_models:
    old_model = old_models[0]

    previous_champion_model_id = old_model.id
    old_metadata = old_model.get_all_metadata()
    previous_promote_task_id = old_metadata.get("promote_task_id", {}).get("value", "")

    old_tags = list(old_model.tags or [])

    old_tags = [tag for tag in old_tags if tag != "champion"]

    if "archived" not in old_tags:
        old_tags.append("archived")

    client.models.edit(
        model=old_model.id,
        tags=old_tags,
    )

    old_model.set_metadata(
        "archived_time",
        str(datetime.now()),
    )

    old_model.set_metadata(
        "archived_by",
        task.id,
    )

    old_model.set_metadata(
        "replaced_by",
        new_model_id,
    )

# =====================================================
# Promote new model
# =====================================================
new_model = Model(
    model_id=new_model_id,
)

new_tags = list(new_model.tags or [])

new_tags = [tag for tag in new_tags if tag != "candidate"]

if "champion" not in new_tags:
    new_tags.append("champion")

client.models.edit(
    model=new_model_id,
    tags=new_tags,
)

new_model.set_metadata(
    "promoted_time",
    str(datetime.now()),
)

new_model.set_metadata(
    "promoted_by",
    task.id,
)

new_model.set_metadata(
    "previous_champion_model_id",
    previous_champion_model_id or "",
)

new_model.set_metadata(
    "previous_promote_task_id",
    previous_promote_task_id or "",
)

new_model.set_metadata(
    "feature_dataset_id",
    compare_lineage["feature_dataset_id"],
)

new_model.set_metadata(
    "train_task_id",
    compare_lineage["train_task_id"],
)

new_model.set_metadata(
    "evaluate_task_id",
    compare_lineage["evaluate_task_id"],
)

new_model.set_metadata(
    "register_task_id",
    compare_lineage["register_task_id"],
)

new_model.set_metadata(
    "compare_task_id",
    compare_lineage["compare_task_id"],
)

promote_summary = {
    "promoted": True,
    "status": "SUCCESS",
    "champion_model_id": new_model_id,
    "previous_champion_model_id": previous_champion_model_id,
    "promotion_time": str(datetime.now()),
}

promote_lineage = {
    "promote_task_id": task.id,
    "compare_task_id": compare_lineage["compare_task_id"],
    "register_task_id": compare_lineage["register_task_id"],
    "train_task_id": compare_lineage["train_task_id"],
    "evaluate_task_id": compare_lineage["evaluate_task_id"],
    "hpo_task_id": compare_lineage["hpo_task_id"],
    "feature_dataset_id": compare_lineage["feature_dataset_id"],
    "champion_model_id": new_model_id,
}

task.upload_artifact(
    "promote_summary",
    promote_summary,
)

task.upload_artifact(
    "promote_lineage",
    promote_lineage,
)

task.get_logger().report_text(
    f"""
        Previous champion model id: {previous_champion_model_id}
        New champion model id: {new_model_id}
    """
)

task.get_logger().report_text(
    f"""
        Promotion Summary

        Status : SUCCESS

        Previous Champion Model ID : {previous_champion_model_id}

        Current Champion Model ID : {new_model_id}

        Promotion Task : {task.id}
    """
)


task.flush()
task.close()
