from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
import time

from clearml.automation import PipelineController
from clearml import Task

from config import (
    CLEARML_SERVER_URL,
    DEPLOYMENT_VERSION,
    PROJECT_PIPELINE,
    SERVICES_QUEUE,
    TRAINING_PIPELINE_NAME,
)
from pipelines.specs import (
    TRAINING_STEPS,
    add_specs_to_pipeline,
    build_pipeline_manifest,
    validate_pipeline_specs,
)

Task.force_requirements_env_freeze(requirements_file="requirements-controller.txt")


def detect_run_mode() -> str:
    """Detect whether this controller was launched by auto-retraining."""

    try:
        from clearml import Task as SingleTask

        current_task = SingleTask.current_task()
        if current_task and current_task.get_parameter(
            "Trigger/auto_retraining_task_id"
        ):
            return "AUTOMATED"
    except Exception:
        pass
    return "MANUAL"


def wait_for_pipeline_start(pipeline_id: str, max_wait_time: int = 30) -> bool:
    """Wait briefly for ClearML to move the controller out of initial states."""

    wait_interval = 2
    elapsed_time = 0

    print(f"\n⏳ Waiting for pipeline to start (max {max_wait_time}s)...")
    while elapsed_time < max_wait_time:
        try:
            from clearml import Task

            current_pipeline_task = Task.get_task(task_id=pipeline_id)
            pipeline_status = current_pipeline_task.get_status()
            print(f"   Pipeline status: {pipeline_status}")

            if pipeline_status not in ["created", "queued"]:
                print(f"✅ Pipeline status confirmed: {pipeline_status}")
                return True

            time.sleep(wait_interval)
            elapsed_time += wait_interval
        except Exception as e:
            print(f"⚠️ Could not fetch pipeline status: {str(e)}")
            time.sleep(wait_interval)
            elapsed_time += wait_interval

    return False


def main() -> None:
    validate_pipeline_specs(TRAINING_STEPS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_mode = detect_run_mode()
    pipe = PipelineController(
        project=PROJECT_PIPELINE,
        name=TRAINING_PIPELINE_NAME,
        version=DEPLOYMENT_VERSION,
    )
    pipe.task.set_script(
        working_dir=".",
        entry_point="pipelines/training_pipeline.py",
    )

    if run_mode == "AUTOMATED":
        pipe.task.set_tags(["pipeline", "training", "automated"])
    else:
        pipe.task.set_tags(["pipeline", "training", "manual"])

    pipe.task.set_comment(
        f"Training Pipeline ({run_mode})\nTimestamp: {timestamp}\nRun Mode: {run_mode}"
    )
    pipe.task.connect(
        {
            "pipeline_type": "training",
            "pipeline_name": TRAINING_PIPELINE_NAME,
            "deployment_version": DEPLOYMENT_VERSION,
            "run_mode": run_mode,
            "timestamp": timestamp,
        },
        name="Pipeline Args",
    )
    pipe.task.upload_artifact(
        "pipeline_manifest",
        build_pipeline_manifest(
            pipeline_type="training",
            pipeline_name=TRAINING_PIPELINE_NAME,
            deployment_version=DEPLOYMENT_VERSION,
            run_mode=run_mode,
            timestamp=timestamp,
            specs=TRAINING_STEPS,
        ),
    )

    pipe.set_default_execution_queue(SERVICES_QUEUE)
    add_specs_to_pipeline(pipe, TRAINING_STEPS)

    pipe.task.flush()

    print("=" * 70)
    print("📌 Starting Training Pipeline...")
    print(f"   Run Mode: {run_mode}")
    print("=" * 70)

    pipe.start(queue=SERVICES_QUEUE)
    pipeline_id = pipe.task.id

    print("=" * 70)
    print("✅ Training Pipeline started")
    print("=" * 70)
    print(f"   Task ID: {pipeline_id}")
    print(f"   Pipeline Name: {TRAINING_PIPELINE_NAME}")
    print(f"   Version: {DEPLOYMENT_VERSION}")
    print(f"   Timestamp: {timestamp}")
    print(f"   UI URL: {CLEARML_SERVER_URL}/tasks/{pipeline_id}")
    print("=" * 70)

    wait_for_pipeline_start(pipeline_id)

    print("\n📤 Finalizing task...")
    pipe.task.flush()

    print("=" * 70)
    print("✅ Training Pipeline configuration completed successfully!")
    print("=" * 70)
    print("   Pipeline will continue running in background")
    print(f"   Monitor progress at: {CLEARML_SERVER_URL}/tasks/{pipeline_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
