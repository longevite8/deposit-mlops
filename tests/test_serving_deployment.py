import argparse
import unittest

from scripts.py.serving.deploy_clearml_serving import endpoint_name
from scripts.py.serving.verify_clearml_serving import endpoint_url
from serving.clearml_preprocess import _normalize_history_frame


class ServingDeploymentTest(unittest.TestCase):
    def test_endpoint_name_uses_prefix_when_endpoint_is_empty(self):
        self.assertEqual(endpoint_name("deposit/cashflow", ""), "deposit_cashflow")

    def test_endpoint_name_prefers_explicit_endpoint(self):
        self.assertEqual(
            endpoint_name("deposit/cashflow", "custom/endpoint"),
            "custom_endpoint",
        )

    def test_endpoint_name_supports_candidate_staging_endpoint(self):
        self.assertEqual(
            endpoint_name("", "deposit/cashflow/candidate"),
            "deposit_cashflow_candidate",
        )

    def test_verify_endpoint_url_includes_version_when_present(self):
        args = argparse.Namespace(
            base_url="http://127.0.0.1:8082",
            endpoint_prefix="deposit/cashflow",
            endpoint="",
            version="production",
        )
        self.assertEqual(
            endpoint_url(args),
            "http://127.0.0.1:8082/serve/deposit_cashflow/production",
        )

    def test_normalize_history_frame_accepts_deposit_rows(self):
        frame = _normalize_history_frame(
            {
                "history": [
                    {"date": "2026-01-01", "cashflow": "100"},
                    {"date": "2026-01-02", "cashflow": 120},
                ]
            }
        )

        self.assertEqual(
            list(frame.columns),
            ["unique_id", "ds", "y", "month", "weekday"],
        )
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["y"].tolist(), [100, 120])

    def test_normalize_history_frame_rejects_missing_target(self):
        with self.assertRaises(ValueError):
            _normalize_history_frame({"history": [{"date": "2026-01-01"}]})


if __name__ == "__main__":
    unittest.main()
