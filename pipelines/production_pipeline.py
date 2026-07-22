from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import time
from typing import Callable

from clearml.automation import PipelineController
from clearml import Model, Task

from config import (
    CLEARML_SERVER_URL,
    DEPLOYMENT_VERSION,
    PRODUCTION_PIPELINE_NAME,
    PROJECT_PIPELINE,
    RUN_PIPELINE_CONTROLLER_LOCALLY,
    SERVICES_QUEUE,
)
from pipelines.specs import (
    PRODUCTION_STEPS,
    add_specs_to_pipeline,
    build_pipeline_manifest,
    validate_pipeline_specs,
)

Task.force_requirements_env_freeze(requirements_file="requirements-controller.txt")


INITIAL_PIPELINE_STATUSES = {"created", "queued"}
RUNNING_PIPELINE_STATUSES = {"in_progress"}
FAILED_PIPELINE_STATUSES = {"failed", "aborted", "stopped"}
SUCCESS_PIPELINE_STATUSES = {"completed", "published"}


def find_published_champion_model(
    model_query: Callable[..., list] | None = None,
):
    """Return the current published champion model, if one is available."""

    model_query = model_query or Model.query_models
    champion_models = model_query(
        tags=["champion"],
        only_published=True,
        max_results=1,
    )
    return champion_models[0] if champion_models else None


def validate_production_prerequisites(
    model_query: Callable[..., list] | None = None,
) -> None:
    """Fail before launching production steps that require a champion model."""

    champion_model = find_published_champion_model(model_query=model_query)
    if champion_model:
        print(f"✅ Published champion model found: {champion_model.id}")
        return

    raise RuntimeError(
        "❌ No published champion model found for production inference.\n"
        "   Run the training pipeline until a candidate passes evaluation, is "
        "registered, and is promoted with the 'champion' tag.\n"
        "   Required ClearML query: tags=['champion'], only_published=True"
    )


def wait_for_pipeline_start(
    pipeline_id: str,
    max_wait_time: int = 30,
    task_getter: Callable[[str], Task] | None = None,
) -> str:
    """Wait briefly for ClearML to move the controller into a useful status."""

    wait_interval = 2
    elapsed_time = 0
    task_getter = task_getter or (lambda task_id: Task.get_task(task_id=task_id))
    last_status = "unknown"

    print(f"\n⏳ Waiting for pipeline to start (max {max_wait_time}s)...")
    while elapsed_time < max_wait_time:
        try:
            current_pipeline_task = task_getter(pipeline_id)
            pipeline_status = current_pipeline_task.get_status()
            last_status = pipeline_status
            print(f"   Pipeline status: {pipeline_status}")

            if pipeline_status in FAILED_PIPELINE_STATUSES:
                print(f"❌ Pipeline failed with status: {pipeline_status}")
                return pipeline_status

            if pipeline_status in RUNNING_PIPELINE_STATUSES | SUCCESS_PIPELINE_STATUSES:
                print(f"✅ Pipeline status confirmed: {pipeline_status}")
                return pipeline_status

            if pipeline_status not in INITIAL_PIPELINE_STATUSES:
                print(f"⚠️ Pipeline status confirmed: {pipeline_status}")
                return pipeline_status

            time.sleep(wait_interval)
            elapsed_time += wait_interval
        except Exception as e:
            print(f"⚠️ Could not fetch pipeline status: {str(e)}")
            time.sleep(wait_interval)
            elapsed_time += wait_interval

    return last_status


def main() -> None:
    validate_pipeline_specs(PRODUCTION_STEPS)
    try:
        validate_production_prerequisites()
    except RuntimeError as e:
        print(str(e))
        raise SystemExit(1) from e

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipe = PipelineController(
        project=PROJECT_PIPELINE,
        name=PRODUCTION_PIPELINE_NAME,
        version=DEPLOYMENT_VERSION,
    )
    pipe.task.set_script(
        working_dir=".",
        entry_point="pipelines/production_pipeline.py",
    )

    pipe.task.set_tags(["pipeline", "production"])
    pipe.task.set_comment(f"Production Pipeline (automated)\nTimestamp: {timestamp}")
    pipe.task.connect(
        {
            "pipeline_type": "production",
            "pipeline_name": PRODUCTION_PIPELINE_NAME,
            "deployment_version": DEPLOYMENT_VERSION,
            "run_mode": "AUTOMATED",
            "timestamp": timestamp,
        },
        name="Pipeline Args",
    )
    pipe.task.upload_artifact(
        "pipeline_manifest",
        build_pipeline_manifest(
            pipeline_type="production",
            pipeline_name=PRODUCTION_PIPELINE_NAME,
            deployment_version=DEPLOYMENT_VERSION,
            run_mode="AUTOMATED",
            timestamp=timestamp,
            specs=PRODUCTION_STEPS,
        ),
    )

    pipe.set_default_execution_queue(SERVICES_QUEUE)
    add_specs_to_pipeline(pipe, PRODUCTION_STEPS)

    pipe.task.flush()

    print("=" * 70)
    print("📌 Starting Production Pipeline...")
    print("=" * 70)

    if RUN_PIPELINE_CONTROLLER_LOCALLY:
        pipe.start_locally(run_pipeline_steps_locally=False)
    else:
        pipe.start(queue=SERVICES_QUEUE)
    pipeline_id = pipe.task.id

    print("=" * 70)
    print("✅ Production Pipeline started")
    print("=" * 70)
    print(f"   Task ID: {pipeline_id}")
    print(f"   Project: {pipe.task.project}")
    print(f"   Pipeline Name: {PRODUCTION_PIPELINE_NAME}")
    print(f"   Version: {DEPLOYMENT_VERSION}")
    print(f"   Timestamp: {timestamp}")
    print(f"   UI URL: {CLEARML_SERVER_URL}/tasks/{pipeline_id}")
    print("=" * 70)

    pipeline_status = wait_for_pipeline_start(pipeline_id)
    if pipeline_status in FAILED_PIPELINE_STATUSES:
        print("\n❌ Production Pipeline failed during startup.")
        print(f"   Controller status: {pipeline_status}")
        print(
            f"   Inspect failed child steps at: {CLEARML_SERVER_URL}/tasks/{pipeline_id}"
        )
        raise SystemExit(1)

    print("\n📤 Finalizing task...")
    pipe.task.flush()

    print("=" * 70)
    print("✅ Production Pipeline configuration completed successfully!")
    print("=" * 70)
    print("   Pipeline will continue running in background")
    print(f"   Monitor progress at: {CLEARML_SERVER_URL}/tasks/{pipeline_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
