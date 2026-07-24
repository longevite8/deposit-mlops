import argparse
import unittest
from unittest.mock import patch

from scripts.py.schedule_production_pipeline import (
    ProductionScheduleConfig,
    bool_from_text,
    config_from_args,
    main,
    register_daily_schedule,
    start_scheduler,
    validate_utc_time,
)


class FakeScheduler:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.add_task_calls = []
        self.started_locally = False
        self.execute_remote_calls = []
        self.serialized = False
        self._task = self
        FakeScheduler.instances.append(self)

    def add_task(self, **kwargs):
        self.add_task_calls.append(kwargs)

    def execute_remotely(self, **kwargs):
        self.execute_remote_calls.append(kwargs)

    def _serialize(self):
        self.serialized = True

    def start(self):
        self.started_locally = True


class FakeLocalTask:
    @staticmethod
    def running_locally():
        return True

    @staticmethod
    def current_task():
        return None


class FakeRemoteTask:
    @staticmethod
    def running_locally():
        return False

    @staticmethod
    def current_task():
        return None


class ProductionSchedulerTest(unittest.TestCase):
    def setUp(self):
        FakeScheduler.instances = []

    def test_register_daily_schedule_uses_clearml_daily_utc_spec(self):
        config = ProductionScheduleConfig(
            pipeline_task_id="production-task-id",
            scheduler_project="Deposit-CashFlow/Pipelines",
            scheduler_task_name="Scheduled Production Pipeline Controller",
            job_name="daily_production_pipeline",
            queue="mco-services",
            target_project="Deposit-CashFlow/Pipelines",
            sync_minutes=5.0,
            utc_hour=19,
            utc_minute=0,
            start_remotely=True,
            execute_immediately=False,
        )

        scheduler = register_daily_schedule(config, FakeScheduler)

        self.assertEqual(
            scheduler.init_kwargs,
            {
                "sync_frequency_minutes": 5.0,
                "force_create_task_project": "Deposit-CashFlow/Pipelines",
                "force_create_task_name": "Scheduled Production Pipeline Controller",
            },
        )
        self.assertEqual(
            scheduler.add_task_calls,
            [
                {
                    "schedule_task_id": "production-task-id",
                    "queue": "mco-services",
                    "name": "daily_production_pipeline",
                    "target_project": "Deposit-CashFlow/Pipelines",
                    "minute": 0,
                    "hour": 19,
                    "day": 1,
                    "single_instance": True,
                    "recurring": True,
                    "execute_immediately": False,
                }
            ],
        )

    def test_start_scheduler_remote_enqueues_scheduler_task_from_local_process(self):
        config = ProductionScheduleConfig(
            pipeline_task_id="production-task-id",
            scheduler_project="Deposit-CashFlow/Pipelines",
            scheduler_task_name="scheduler",
            job_name="daily",
            queue="mco-services",
            target_project="Deposit-CashFlow/Pipelines",
            sync_minutes=5.0,
            utc_hour=19,
            utc_minute=0,
            start_remotely=True,
            execute_immediately=False,
        )
        scheduler = FakeScheduler()

        mode = start_scheduler(scheduler, config, task_cls=FakeLocalTask)

        self.assertEqual(mode, "remote")
        self.assertEqual(
            scheduler.execute_remote_calls,
            [{"queue_name": "mco-services", "exit_process": True}],
        )
        self.assertTrue(scheduler.serialized)
        self.assertFalse(scheduler.started_locally)

    def test_start_scheduler_remote_controller_starts_loop_without_reenqueue(self):
        config = ProductionScheduleConfig(
            pipeline_task_id="production-task-id",
            scheduler_project="Deposit-CashFlow/Pipelines",
            scheduler_task_name="scheduler",
            job_name="daily",
            queue="mco-services",
            target_project="Deposit-CashFlow/Pipelines",
            sync_minutes=5.0,
            utc_hour=19,
            utc_minute=0,
            start_remotely=True,
            execute_immediately=False,
        )
        scheduler = FakeScheduler()

        mode = start_scheduler(scheduler, config, task_cls=FakeRemoteTask)

        self.assertEqual(mode, "remote")
        self.assertTrue(scheduler.started_locally)
        self.assertEqual(scheduler.execute_remote_calls, [])

    def test_start_scheduler_local_blocks_in_local_mode(self):
        config = ProductionScheduleConfig(
            pipeline_task_id="production-task-id",
            scheduler_project="Deposit-CashFlow/Pipelines",
            scheduler_task_name="scheduler",
            job_name="daily",
            queue="mco-services",
            target_project="Deposit-CashFlow/Pipelines",
            sync_minutes=5.0,
            utc_hour=19,
            utc_minute=0,
            start_remotely=False,
            execute_immediately=False,
        )
        scheduler = FakeScheduler()

        mode = start_scheduler(scheduler, config)

        self.assertEqual(mode, "local")
        self.assertTrue(scheduler.started_locally)
        self.assertEqual(scheduler.execute_remote_calls, [])

    def test_config_requires_production_pipeline_task_id(self):
        args = argparse.Namespace(
            pipeline_task_id="",
            scheduler_project="Deposit-CashFlow/Pipelines",
            scheduler_task_name="scheduler",
            job_name="daily",
            queue="mco-services",
            target_project="Deposit-CashFlow/Pipelines",
            sync_minutes=5.0,
            utc_hour=19,
            utc_minute=0,
            start_remotely=True,
            execute_immediately=False,
        )

        with self.assertRaises(ValueError) as raised:
            config_from_args(args)

        self.assertIn("PRODUCTION_PIPELINE_ID", str(raised.exception))

    def test_validate_utc_time_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            validate_utc_time(24, 0)
        with self.assertRaises(ValueError):
            validate_utc_time(23, 60)

    def test_bool_from_text_accepts_common_values(self):
        self.assertTrue(bool_from_text("true"))
        self.assertTrue(bool_from_text("1"))
        self.assertFalse(bool_from_text("false"))
        self.assertFalse(bool_from_text("0"))

    def test_main_freezes_controller_requirements_before_scheduler_task_creation(self):
        with (
            patch(
                "scripts.py.schedule_production_pipeline.build_parser"
            ) as build_parser,
            patch("clearml.Task.force_requirements_env_freeze") as freeze,
            patch(
                "scripts.py.schedule_production_pipeline.register_daily_schedule"
            ) as register,
            patch(
                "scripts.py.schedule_production_pipeline.start_scheduler",
                return_value="remote",
            ),
        ):
            build_parser.return_value.parse_args.return_value = argparse.Namespace(
                pipeline_task_id="production-task-id",
                scheduler_project="Deposit-CashFlow/Pipelines",
                scheduler_task_name="scheduler",
                job_name="daily",
                queue="mco-services",
                target_project="Deposit-CashFlow/Pipelines",
                sync_minutes=5.0,
                utc_hour=19,
                utc_minute=0,
                start_remotely=True,
                execute_immediately=False,
            )

            main()

        freeze.assert_called_once_with(requirements_file="requirements-controller.txt")
        self.assertTrue(register.called)


if __name__ == "__main__":
    unittest.main()
