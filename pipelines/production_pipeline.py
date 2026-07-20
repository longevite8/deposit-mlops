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
    PRODUCTION_PIPELINE_NAME,
    PROJECT_PIPELINE,
    SERVICES_QUEUE,
)
from pipelines.specs import (
    PRODUCTION_STEPS,
    add_specs_to_pipeline,
    build_pipeline_manifest,
    validate_pipeline_specs,
)

Task.force_requirements_env_freeze(requirements_file="requirements-controller.txt")


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

            if pipeline_status not in ["created", "queued", "in_progress"]:
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
    validate_pipeline_specs(PRODUCTION_STEPS)

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

    wait_for_pipeline_start(pipeline_id)

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
