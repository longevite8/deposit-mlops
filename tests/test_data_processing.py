import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from business.data_processing import (
    DataExtractionConfigError,
    DataExtractionDatabaseError,
    DataExtractionNotFoundError,
    SourceConfig,
    build_cashflow_query,
    build_extract_source_summary,
    extract_data,
    normalize_cashflow_frame,
    validate_source_config,
)


def valid_source_config(**overrides):
    values = {
        "db_user": "vega",
        "db_password": "secret",
        "db_host": "localhost",
        "db_port": 5432,
        "db_name": "vc_mco",
        "db_schema": "public",
        "project_name": "Deposit",
        "flow_type": "Vào",
        "approval_status": "Đã duyệt",
        "from_date": "2025-01-01",
        "to_date": "2025-01-31",
    }
    values.update(overrides)
    return SourceConfig(**values)


class DataProcessingTest(unittest.TestCase):
    def test_build_cashflow_query_uses_expected_filters(self):
        config = valid_source_config(project_name="Deposit Portfolio")

        query, params = build_cashflow_query(config)

        self.assertIn("public.project_real_cashflow", query)
        self.assertIn("SUM(prc.amount)::float AS cashflow", query)
        self.assertEqual(
            params,
            {
                "project_name": "Deposit Portfolio",
                "flow_type": "Vào",
                "approval_status": "Đã duyệt",
                "from_date": "2025-01-01",
                "to_date": "2025-01-31",
            },
        )

    def test_normalize_cashflow_frame_aggregates_and_sorts_daily_values(self):
        raw_df = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-01", "2025-01-01"],
                "cashflow": ["7.5", "10", "5"],
            }
        )

        normalized = normalize_cashflow_frame(raw_df)

        self.assertEqual(
            normalized["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2025-01-01", "2025-01-02"],
        )
        self.assertEqual(normalized["cashflow"].tolist(), [15.0, 7.5])

    def test_validate_source_config_rejects_missing_required_values(self):
        config = valid_source_config(project_name="")

        with self.assertRaises(DataExtractionConfigError) as raised:
            validate_source_config(config)

        self.assertIn("SOURCE_PROJECT_NAME", str(raised.exception))

    def test_validate_source_config_rejects_unsafe_schema_identifier(self):
        config = valid_source_config(db_schema="public;drop table project")

        with self.assertRaises(DataExtractionConfigError):
            validate_source_config(config)

    @patch("business.data_processing.pd.read_sql_query")
    @patch("sqlalchemy.create_engine")
    def test_extract_data_raises_not_found_for_empty_query_result(
        self, create_engine, read_sql_query
    ):
        connection = MagicMock()
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = connection
        create_engine.return_value = engine
        read_sql_query.return_value = pd.DataFrame(columns=["date", "cashflow"])

        with self.assertRaises(DataExtractionNotFoundError):
            extract_data(valid_source_config())

    @patch("sqlalchemy.create_engine")
    def test_extract_data_preserves_database_failure_as_cause(self, create_engine):
        database_error = RuntimeError("connection refused")
        create_engine.side_effect = database_error

        with self.assertRaises(DataExtractionDatabaseError) as raised:
            extract_data(valid_source_config())

        self.assertIs(raised.exception.__cause__, database_error)

    def test_build_extract_source_summary_reports_source_window(self):
        config = valid_source_config()
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-03"]),
                "cashflow": [10.0, 20.0],
            }
        )

        summary = build_extract_source_summary(config, df)

        self.assertEqual(summary["source_type"], "postgresql")
        self.assertEqual(summary["source_project_name"], "Deposit")
        self.assertEqual(summary["source_from_date"], "2025-01-01")
        self.assertEqual(summary["source_to_date"], "2025-01-31")
        self.assertEqual(summary["min_date"], "2025-01-01")
        self.assertEqual(summary["max_date"], "2025-01-03")


if __name__ == "__main__":
    unittest.main()
