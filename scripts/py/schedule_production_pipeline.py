"""Register a recurring ClearML schedule for the production pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from config import (
    PRODUCTION_PIPELINE_ID,
    PRODUCTION_SCHEDULER_DAILY_UTC_HOUR,
    PRODUCTION_SCHEDULER_DAILY_UTC_MINUTE,
    PRODUCTION_SCHEDULER_EXECUTE_IMMEDIATELY,
    PRODUCTION_SCHEDULER_JOB_NAME,
    PRODUCTION_SCHEDULER_PROJECT,
    PRODUCTION_SCHEDULER_QUEUE,
    PRODUCTION_SCHEDULER_START_REMOTELY,
    PRODUCTION_SCHEDULER_SYNC_MINUTES,
    PRODUCTION_SCHEDULER_TARGET_PROJECT,
    PRODUCTION_SCHEDULER_TASK_NAME,
    PROJECT_PIPELINE,
    SERVICES_QUEUE,
)


@dataclass(frozen=True)
class ProductionScheduleConfig:
    """Runtime settings for one recurring production pipeline schedule."""

    pipeline_task_id: str
    scheduler_project: str
    scheduler_task_name: str
    job_name: str
    queue: str
    target_project: str
    sync_minutes: float
    utc_hour: int
    utc_minute: int
    start_remotely: bool
    execute_immediately: bool


def bool_from_text(value: str | bool) -> bool:
    """Parse common CLI/env boolean values."""

    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def validate_utc_time(hour: int, minute: int) -> None:
    """Validate ClearML scheduler UTC time components."""

    if not 0 <= hour <= 23:
        raise ValueError(f"UTC hour must be between 0 and 23, got {hour}")
    if not 0 <= minute <= 59:
        raise ValueError(f"UTC minute must be between 0 and 59, got {minute}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register the ClearML daily schedule for Production Pipeline."
    )
    parser.add_argument(
        "--pipeline_task_id",
        default=PRODUCTION_PIPELINE_ID,
        help="ClearML task ID of the Production Pipeline controller to clone daily.",
    )
    parser.add_argument("--scheduler_project", default=PRODUCTION_SCHEDULER_PROJECT)
    parser.add_argument("--scheduler_task_name", default=PRODUCTION_SCHEDULER_TASK_NAME)
    parser.add_argument("--job_name", default=PRODUCTION_SCHEDULER_JOB_NAME)
    parser.add_argument("--queue", default=PRODUCTION_SCHEDULER_QUEUE)
    parser.add_argument("--target_project", default=PRODUCTION_SCHEDULER_TARGET_PROJECT)
    parser.add_argument(
        "--sync_minutes",
        type=float,
        default=PRODUCTION_SCHEDULER_SYNC_MINUTES,
        help="How often the scheduler controller syncs its ClearML state.",
    )
    parser.add_argument(
        "--utc_hour",
        type=int,
        default=PRODUCTION_SCHEDULER_DAILY_UTC_HOUR,
        help="UTC hour for the daily run. ClearML TaskScheduler uses UTC.",
    )
    parser.add_argument(
        "--utc_minute",
        type=int,
        default=PRODUCTION_SCHEDULER_DAILY_UTC_MINUTE,
        help="UTC minute for the daily run. ClearML TaskScheduler uses UTC.",
    )
    parser.add_argument(
        "--start_remotely",
        type=bool_from_text,
        default=PRODUCTION_SCHEDULER_START_REMOTELY,
        help="Start the scheduler controller on the selected ClearML queue.",
    )
    parser.add_argument(
        "--execute_immediately",
        type=bool_from_text,
        default=PRODUCTION_SCHEDULER_EXECUTE_IMMEDIATELY,
        help="Run one production pipeline clone immediately after registration.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ProductionScheduleConfig:
    validate_utc_time(args.utc_hour, args.utc_minute)
    pipeline_task_id = str(args.pipeline_task_id or "").strip()
    if not pipeline_task_id:
        raise ValueError(
            "Missing production pipeline task ID. Set PRODUCTION_PIPELINE_ID or pass "
            "--pipeline_task_id with a ClearML Production Pipeline controller task ID."
        )
    return ProductionScheduleConfig(
        pipeline_task_id=pipeline_task_id,
        scheduler_project=args.scheduler_project or PROJECT_PIPELINE,
        scheduler_task_name=args.scheduler_task_name
        or PRODUCTION_SCHEDULER_TASK_NAME,
        job_name=args.job_name or "daily_production_pipeline",
        queue=args.queue or SERVICES_QUEUE,
        target_project=args.target_project or PROJECT_PIPELINE,
        sync_minutes=float(args.sync_minutes),
        utc_hour=int(args.utc_hour),
        utc_minute=int(args.utc_minute),
        start_remotely=bool(args.start_remotely),
        execute_immediately=bool(args.execute_immediately),
    )


def register_daily_schedule(config: ProductionScheduleConfig, scheduler_cls: Any):
    """Create the ClearML scheduler and register the daily production job."""

    scheduler = scheduler_cls(
        sync_frequency_minutes=config.sync_minutes,
        force_create_task_project=config.scheduler_project,
        force_create_task_name=config.scheduler_task_name,
    )
    scheduler.add_task(
        schedule_task_id=config.pipeline_task_id,
        queue=config.queue,
        name=config.job_name,
        target_project=config.target_project,
        minute=config.utc_minute,
        hour=config.utc_hour,
        day=1,
        single_instance=True,
        recurring=True,
        execute_immediately=config.execute_immediately,
    )
    return scheduler


def start_scheduler(
    scheduler,
    config: ProductionScheduleConfig,
    task_cls: Any | None = None,
) -> str:
    """Start the scheduler controller and return the selected mode."""

    if task_cls is None:
        from clearml import Task

        task_cls = Task

    if config.start_remotely:
        if task_cls.running_locally():
            scheduler_task = getattr(scheduler, "_task", None) or task_cls.current_task()
            if scheduler_task is None:
                raise RuntimeError("Could not find ClearML scheduler task to enqueue.")
            scheduler._serialize()
            scheduler_task.execute_remotely(
                queue_name=config.queue,
                exit_process=True,
            )
            return "remote"
        scheduler.start()
        return "remote"
    scheduler.start()
    return "local"


def main() -> None:
    from clearml import Task
    from clearml.automation import TaskScheduler

    Task.force_requirements_env_freeze(requirements_file="requirements-controller.txt")
    config = config_from_args(build_parser().parse_args())
    scheduler = register_daily_schedule(config, TaskScheduler)
    print(
        "Registered daily Production Pipeline schedule: "
        f"task_id={config.pipeline_task_id}, queue={config.queue}, "
        f"time={config.utc_hour:02d}:{config.utc_minute:02d} UTC, "
        f"single_instance=true"
    )
    mode = start_scheduler(scheduler, config)
    print(f"Scheduler controller started in {mode} mode.")


if __name__ == "__main__":
    main()
