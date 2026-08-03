from clearml import (
    Model,
    Task,
)
from clearml.backend_api.session.client import APIClient

from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_REGISTER_NAME,
)
from helpers import wait_for_artifact, wait_for_metadata

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

task.get_logger().report_text("📍 Loading hpo_task_id from train task...")

# Try to get hpo_task_id from train task artifacts (preferred)
try:
    hpo_task_id = wait_for_artifact(
        train_task,
        "hpo_task_id",
        max_retries=5,
        wait_interval=1.0,
        logger_obj=task,
    )
    task.get_logger().report_text(f"✅ Got hpo_task_id from artifact: {hpo_task_id}")
except Exception as e:
    # Fallback: Try to get from parameters
    task.get_logger().report_text(
        f"⚠️ hpo_task_id artifact not found ({e}), checking parameters..."
    )
    train_params = train_task.get_parameters()
    hpo_task_id = train_params.get("General/hpo_task_id") or train_params.get(
        "General/compare_hpo_task_id"
    )
    if hpo_task_id:
        task.get_logger().report_text(
            f"✅ Got hpo_task_id from parameters: {hpo_task_id}"
        )
    else:
        task.get_logger().report_text(
            "⚠️ hpo_task_id not found in artifact or parameters"
        )
        hpo_task_id = None

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
        "compare_hpo_task_id",
        str(compare_hpo_task_id) if compare_hpo_task_id else "unknown",
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
    "compare_hpo_task_id": compare_hpo_task_id,  # Có thể None, nhưng vẫn track
    "model_id": model_id,
    "feature_dataset_id": feature_dataset_id,
}

# ✅ Upload compare_hpo_task_id artifact for downstream tasks
if compare_hpo_task_id:
    task.upload_artifact("compare_hpo_task_id", compare_hpo_task_id)
    task.get_logger().report_text(
        f"📍 Uploaded compare_hpo_task_id: {compare_hpo_task_id}"
    )

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
