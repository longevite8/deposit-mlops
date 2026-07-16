import tempfile
import unittest
from pathlib import Path
import zipfile

import pandas as pd

from business.forecasting import (
    archive_model_dir,
    calculate_forecast_metrics,
    historical_exogenous_columns,
    prepare_forecast_frame,
)


class ForecastingTest(unittest.TestCase):
    def test_prepare_forecast_frame_converts_and_aggregates_deposit_schema(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-01", "2025-01-01"],
                "cashflow": ["7.5", "10", "5"],
                "lag_1": [1.0, 2.0, 3.0],
            }
        )

        forecast_df = prepare_forecast_frame(df, unique_id="Deposit_Test")

        self.assertEqual(
            list(forecast_df.columns),
            ["unique_id", "ds", "y", "month", "weekday"],
        )
        self.assertEqual(forecast_df["unique_id"].unique().tolist(), ["Deposit_Test"])
        self.assertEqual(
            forecast_df["ds"].dt.strftime("%Y-%m-%d").tolist(),
            ["2025-01-01", "2025-01-02"],
        )
        self.assertEqual(forecast_df["y"].tolist(), [15.0, 7.5])
        self.assertEqual(forecast_df["month"].tolist(), [1, 1])

    def test_historical_exogenous_columns_returns_calendar_features(self):
        df = pd.DataFrame(
            {
                "unique_id": ["Deposit"],
                "ds": pd.to_datetime(["2025-01-01"]),
                "y": [10.0],
                "month": [1],
                "weekday": [2],
            }
        )

        self.assertEqual(historical_exogenous_columns(df), ["month", "weekday"])

    def test_calculate_forecast_metrics_adds_legacy_quality_gate_aliases(self):
        forecast_df = pd.DataFrame(
            {
                "unique_id": ["Deposit", "Deposit"],
                "ds": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "y": [100.0, 200.0],
                "month": [1, 1],
                "weekday": [2, 3],
                "NHITS": [90.0, 210.0],
                "NBEATSx": [80.0, 220.0],
            }
        )

        metrics = calculate_forecast_metrics(forecast_df)

        self.assertEqual(metrics["primary_model"], "NHITS")
        self.assertAlmostEqual(metrics["mape"], 0.075)
        self.assertIn("NHITS_rmse", metrics)
        self.assertIn("NBEATSx_mape", metrics)
        self.assertNotIn("month_mape", metrics)
        self.assertNotIn("weekday_mape", metrics)

    def test_archive_model_dir_zips_checkpoint_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "checkpoints"
            model_dir.mkdir()
            (model_dir / "configuration.pkl").write_text("config", encoding="utf-8")
            nested = model_dir / "NHITS"
            nested.mkdir()
            (nested / "model.ckpt").write_text("weights", encoding="utf-8")

            archive_path = Path(tmp) / "model.zip"

            archive_model_dir(model_dir, archive_path)

            with zipfile.ZipFile(archive_path, "r") as zf:
                self.assertEqual(
                    sorted(zf.namelist()),
                    ["NHITS/model.ckpt", "configuration.pkl"],
                )


if __name__ == "__main__":
    unittest.main()
