from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path = [p for p in sys.path if not (p == "/vc-mco" or p.startswith("/vc-mco/"))]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

import numpy as np
import pandas as pd
from pathlib import Path

from clearml import (
    Dataset,
    Model,
    Task,
)

from config import (
    PROJECT_TEMPLATE,
    PROJECT_DATASET,
    TEMPLATE_DRIFT_NAME,
)

from helpers import wait_for_artifact, wait_for_metadata
from business.drift import monitor_drift

logger = logging.getLogger(__name__)

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_DRIFT_NAME,
    task_type=Task.TaskTypes.qc,
)


# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "feature_task_id": "",
    }
)


# =====================================================
# Template creation mode
# =====================================================

if not params["feature_task_id"]:
    task.get_logger().report_text("Template creation mode.")

    task.close()

    raise SystemExit(0)


# =====================================================
# Load current dataset
# =====================================================

feature_task = Task.get_task(task_id=params["feature_task_id"])

# SỬA: Dùng wait_for_artifact để chắc chắn dataset ID sẵn sàng
feature_dataset_id = wait_for_artifact(
    feature_task,
    "feature_dataset_id",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)

feature_dataset = Dataset.get(dataset_id=feature_dataset_id)

local_path = Path(feature_dataset.get_local_copy())

current_train_df = pd.read_parquet(local_path / "train.parquet")

# =====================================================
# Load reference dataset from champion model
# If no champion exists yet (first training run),
# use the current dataset as both reference and current
# so drift detection is skipped gracefully (PASS).
# =====================================================

# Lấy Champion Model metadata (nếu có)
# Ghi chú: xử lý an toàn để đảm bảo reference_df và current_df luôn được khởi tạo
try:
    champion_models = Model.query_models(
        tags=["champion"],
        only_published=True,
        max_results=1,
    )
except Exception as e:
    task.get_logger().report_text(f"Error querying champion models: {e}")
    champion_models = []

if champion_models:
    champion_model = champion_models[0]

    # SỬA: Dùng wait_for_metadata để chắc chắn metadata sẵn sàng (nếu có)
    reference_feature_dataset_id = None
    try:
        reference_feature_dataset_id = wait_for_metadata(
            champion_model,
            "feature_dataset_id",
            max_retries=10,
            wait_interval=2.0,
            logger_obj=task,
        )
    except Exception:
        # ignore and try to read directly from model metadata
        reference_feature_dataset_id = None

    # Fallback to model metadata if wait_for_metadata did not return a valid id
    if not reference_feature_dataset_id or not isinstance(
        reference_feature_dataset_id, str
    ):
        reference_feature_dataset_id = champion_model.get_metadata("feature_dataset_id")

    # If still missing, fallback to dataset by name
    if not reference_feature_dataset_id or not isinstance(
        reference_feature_dataset_id, str
    ):
        task.get_logger().report_text(
            "Champion missing or invalid feature_dataset_id metadata. Falling back to dataset by name."
        )
        champion_dataset = Dataset.get(
            dataset_project=PROJECT_DATASET,
            dataset_name="Deposit Feature Dataset",
        )
        reference_feature_dataset_id = champion_dataset.id
    else:
        champion_dataset = Dataset.get(dataset_id=reference_feature_dataset_id)

    # Load reference dataset dataframe
    try:
        champion_local_path = Path(champion_dataset.get_local_copy())
        reference_df = (
            pd.read_parquet(champion_local_path / "train.parquet")
            .copy()
            .reset_index(drop=True)
        )
    except Exception as e:
        task.get_logger().report_text(
            f"Failed to load champion dataset locally: {e}; falling back to current dataset."
        )
        reference_df = current_train_df.copy().reset_index(drop=True)

    champion_model_id = champion_model.id
    champion_train_task_id = champion_model.get_metadata("train_task_id") or ""

    # =====================================================
    # Current dataset (use test split for production drift)
    # =====================================================
    current_df = current_train_df.copy().reset_index(drop=True)
else:
    task.get_logger().report_text(
        "No champion model found. Using current dataset as reference (first run). "
        "Drift detection will report PASS."
    )

    reference_feature_dataset_id = feature_dataset_id

    champion_model_id = ""

    champion_train_task_id = ""

    reference_df = current_train_df.copy().reset_index(drop=True)

    # =====================================================
    # Current dataset (use test split for production drift)
    # =====================================================

    current_df = current_train_df.copy().reset_index(drop=True)


# =====================================================
# BUSINESS LOGIC: BEGIN
# =====================================================

drift_result = monitor_drift(reference_df, current_df)

# =====================================================
# BUSINESS LOGIC: END
# =====================================================


for drift_feature in drift_result["drift_data"]:
    task.get_logger().report_scalar(
        title="KS Statistic",
        series=drift_feature["feature"],
        value=drift_feature["ks_statistic"],
        iteration=0,
    )

    task.get_logger().report_scalar(
        title="P Value",
        series=drift_feature["feature"],
        value=drift_feature["p_value"],
        iteration=0,
    )

    # ------------------------------------------------
    # Histogram for drift features only
    # ------------------------------------------------

    if drift_feature["drift"]:
        train_hist, bins = np.histogram(
            reference_df[drift_feature["feature"]],
            bins=20,
        )

        test_hist, _ = np.histogram(
            current_df[drift_feature["feature"]],
            bins=bins,
        )

        task.get_logger().report_histogram(
            title=f"{drift_feature['feature']} distribution",
            series="reference",
            values=train_hist,
            iteration=0,
        )

        task.get_logger().report_histogram(
            title=f"{drift_feature['feature']} distribution",
            series="current",
            values=test_hist,
            iteration=0,
        )


# =====================================================
# Drift summary
# =====================================================

drift_table = pd.DataFrame(drift_result["drift_data"]).sort_values("p_value")

drift_summary = {
    "status": drift_result["status"],
    "drift_ratio": drift_result["drift_ratio"],
    "n_features": drift_result["n_features"],
    "n_drift_features": drift_result["n_drift_features"],
    "drift_details": drift_result["drift_data"],
}
drift_lineage = {
    "drift_task_id": task.id,
    "feature_task_id": params["feature_task_id"],
    "feature_dataset_id": feature_dataset_id,
}
task.upload_artifact("drift_summary", drift_summary)
task.upload_artifact("drift_lineage", drift_lineage)

# =====================================================
# Log
# =====================================================

task.get_logger().report_text(
    f"""
    Drift Summary

    Status           : {drift_result["status"]}
    Number of features      : {drift_result["n_features"]}
    Number of drift features: {drift_result["n_drift_features"]}
    Drift ratio      : {drift_result["drift_ratio"]:.2f}
    
    Current Dataset  : {feature_dataset_id}
    Reference Dataset: {reference_feature_dataset_id}
    Champion Model   : {champion_model_id}
    """
)

task.get_logger().report_table(
    title="Drift Table",
    series="features",
    iteration=0,
    table_plot=drift_table,
)

task.get_logger().report_single_value(
    "drift_ratio",
    drift_result["drift_ratio"],
)

task.get_logger().report_single_value(
    "n_drift_features",
    drift_result["n_drift_features"],
)

print(
    "Drift summary:",
    drift_summary,
)

task.flush()
task.close()
