import unittest

from pipelines.production_pipeline import (
    find_published_champion_model,
    validate_production_prerequisites,
    wait_for_pipeline_start,
)


class FakeTask:
    def __init__(self, status):
        self._status = status

    def get_status(self):
        return self._status


class FakeModel:
    id = "champion-model-id"


class PipelineStatusTest(unittest.TestCase):
    def test_find_published_champion_model_uses_production_inference_contract(self):
        calls = []

        def model_query(**kwargs):
            calls.append(kwargs)
            return [FakeModel()]

        champion = find_published_champion_model(model_query=model_query)

        self.assertIsInstance(champion, FakeModel)
        self.assertEqual(
            calls,
            [
                {
                    "tags": ["champion"],
                    "only_published": True,
                    "max_results": 1,
                }
            ],
        )

    def test_validate_prerequisites_fails_when_no_published_champion_exists(self):
        with self.assertRaises(RuntimeError) as raised:
            validate_production_prerequisites(model_query=lambda **kwargs: [])

        self.assertIn("No published champion model", str(raised.exception))

    def test_validate_prerequisites_accepts_published_champion(self):
        validate_production_prerequisites(model_query=lambda **kwargs: [FakeModel()])

    def test_wait_returns_failed_status_immediately(self):
        calls = []

        def task_getter(task_id):
            calls.append(task_id)
            return FakeTask("failed")

        status = wait_for_pipeline_start(
            "pipeline-id",
            max_wait_time=30,
            task_getter=task_getter,
        )

        self.assertEqual(status, "failed")
        self.assertEqual(calls, ["pipeline-id"])

    def test_wait_returns_running_status_when_pipeline_starts(self):
        status = wait_for_pipeline_start(
            "pipeline-id",
            max_wait_time=30,
            task_getter=lambda task_id: FakeTask("in_progress"),
        )

        self.assertEqual(status, "in_progress")

    def test_wait_returns_last_status_on_timeout(self):
        status = wait_for_pipeline_start(
            "pipeline-id",
            max_wait_time=1,
            task_getter=lambda task_id: FakeTask("queued"),
        )

        self.assertEqual(status, "queued")


if __name__ == "__main__":
    unittest.main()
