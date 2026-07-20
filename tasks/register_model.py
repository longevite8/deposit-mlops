from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clearml import (
    Model,
    Task,
)
from clearml.backend_api.session.client import APIClient

from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_REGISTER_NAME,
)

from helpers import wait_for_artifact, wait_for_metadata  # THÊM: Import từ helper

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_REGISTER_NAME,
    task_type=Task.TaskTypes.application,
)


# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "train_task_id": "",
        "evaluate_task_id": "",
    }
)


# =====================================================
# Template creation mode
# =====================================================

if not params["train_task_id"] or not params["evaluate_task_id"]:
    task.get_logger().report_text("Template creation mode.")

    task.close()
    raise SystemExit(0)


# =====================================================
# Load evaluation result
# =====================================================

evaluate_task = Task.get_task(task_id=params["evaluate_task_id"])

evaluate_summary = wait_for_artifact(
    evaluate_task,
    "evaluate_summary",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

# =====================================================
# Load model id
# =====================================================

train_task = Task.get_task(
    task_id=params["train_task_id"],
)

model_id = wait_for_artifact(
    train_task,
    "model_id",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

registered_model = Model(model_id=model_id)

metadata = registered_model.get_all_metadata()

# Dùng wait_for_metadata để chắc chắn metadata sẵn sàng
feature_dataset_id = wait_for_metadata(
    metadata,
    "feature_dataset_id",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

# =====================================================
# Load lineage information
# =====================================================

train_params = train_task.get_parameters()

hpo_task_id = train_params.get("General/hpo_task_id")

# =====================================================
# Publish model
# =====================================================

published = evaluate_summary["passed"]
if published:
    mape = evaluate_summary["mape"]
    r2 = evaluate_summary["r2"]

    registered_model.set_metadata(
        "mape",
        str(mape),
    )

    registered_model.set_metadata(
        "r2",
        str(r2),
    )

    registered_model.set_metadata(
        "feature_dataset_id",
        feature_dataset_id,
    )

    # =====================================================
    # Model lineage
    # =====================================================

    registered_model.set_metadata(
        "hpo_task_id",
        str(hpo_task_id),
    )

    registered_model.set_metadata(
        "train_task_id",
        str(train_task.id),
    )

    registered_model.set_metadata(
        "evaluate_task_id",
        str(evaluate_task.id),
    )

    registered_model.set_metadata(
        "register_task_id",
        str(task.id),
    )

    client = APIClient()

    new_tags = list(registered_model.tags or [])

    if "candidate" not in new_tags:
        new_tags.append("candidate")

    client.models.edit(
        model=registered_model.id,
        tags=new_tags,
    )

    registered_model.publish()

    register_summary = {
        "published": True,
        "model_id": registered_model.id,
        "train_task_id": train_task.id,
        "feature_dataset_id": feature_dataset_id,
        "mape": mape,
        "r2": r2,
    }

    task.get_logger().report_text(f"Published model: {model_id}")

    print(
        "Published:",
        model_id,
    )

else:
    register_summary = {
        "published": False,
        "model_id": None,
        "train_task_id": None,
        "feature_dataset_id": feature_dataset_id,
        "mape": evaluate_summary["mape"],
        "r2": evaluate_summary["r2"],
    }

    task.get_logger().report_text("Quality gate not passed. Keep current champion.")

    print("Skip publish.")

register_lineage = {
    "register_task_id": task.id,
    "train_task_id": train_task.id,
    "evaluate_task_id": evaluate_task.id,
    "hpo_task_id": hpo_task_id,
    "model_id": model_id,
    "feature_dataset_id": feature_dataset_id,
}

task.upload_artifact(
    "register_summary",
    register_summary,
)

task.upload_artifact(
    "register_lineage",
    register_lineage,
)

task.get_logger().report_text(f"Feature Dataset = {feature_dataset_id}")

task.flush()
task.close()
