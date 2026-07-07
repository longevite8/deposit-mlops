import logging

import numpy as np
import pandas as pd
from pathlib import Path

from clearml import (
    Dataset,
    Model,
    Task,
)
from scipy.stats import ks_2samp

from config import (
    DRIFT_PVALUE_THRESHOLD,
    DRIFT_RATIO_THRESHOLD,
    FEATURE_COLUMNS,
    PROJECT_TEMPLATE,
    PROJECT_DATASET,  # THÊM: Import PROJECT_DATASET để dùng cho fallback
    TEMPLATE_DRIFT_NAME,
)

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

feature_dataset_id = feature_task.artifacts["feature_dataset_id"].get()

feature_dataset = Dataset.get(
    dataset_id=feature_dataset_id,
)

local_path = Path(feature_dataset.get_local_copy())

current_train_df = pd.read_parquet(local_path / "train.parquet")

# =====================================================
# Load reference dataset from champion model
# If no champion exists yet (first training run),
# use the current dataset as both reference and current
# so drift detection is skipped gracefully (PASS).
# =====================================================

champion_models = Model.query_models(
    tags=["champion"],
    only_published=True,
    max_results=1,
)

if not champion_models:
    task.get_logger().report_text(
        "No champion model found. Using current dataset as reference (first run). "
        "Drift detection will report PASS."
    )

    reference_feature_dataset_id = feature_dataset_id

    champion_model_id = ""

    champion_train_task_id = ""

    reference_df = current_train_df.copy().reset_index(drop=True)

else:
    champion_model = champion_models[0]
    champion_metadata = champion_model.get_all_metadata()
    champion_model_id = champion_model.id

    # Metadata key is stored as "train_task_id" (set by promote_champion.py)
    champion_train_task_id = champion_metadata.get("train_task_id", {}).get("value", "")

    # SỬA: Truy cập metadata an toàn hơn
    reference_feature_dataset_id = champion_metadata.get("feature_dataset_id", {}).get(
        "value"
    )

    if not reference_feature_dataset_id or not isinstance(
        reference_feature_dataset_id, str
    ):
        task.get_logger().report_text(
            "Champion missing or invalid feature_dataset_id metadata. Falling back to name."
        )
        champion_dataset = Dataset.get(
            dataset_project=PROJECT_DATASET,
            dataset_name="Deposit Feature Dataset",
        )
    else:
        # SỬA: Bọc Dataset.get để tránh crash do Metadata chưa đồng bộ (Race Condition)
        try:
            champion_dataset = Dataset.get(dataset_id=reference_feature_dataset_id)
        except Exception as e:
            task.get_logger().report_text(
                f"Warning: Could not get Dataset by ID {reference_feature_dataset_id}: {e}"
            )
            task.get_logger().report_text(
                "Falling back to latest finalized dataset by name..."
            )
            # Bây giờ PROJECT_DATASET đã được định nghĩa nhờ vào import ở trên
            champion_dataset = Dataset.get(
                dataset_project=PROJECT_DATASET,
                dataset_name="Deposit Feature Dataset",
            )

    reference_feature_dataset_id = champion_dataset.id  # Cập nhật lại ID thực tế
    champion_path = Path(champion_dataset.get_local_copy())

    reference_df = pd.read_parquet(champion_path / "train.parquet").reset_index(
        drop=True
    )

# =====================================================
# Current dataset (use test split for production drift)
# =====================================================

current_df = current_train_df.copy().reset_index(drop=True)


# =====================================================
# Drift detection (Kolmogorov–Smirnov test)
# =====================================================
drift_result = {}

drift_rows = []

n_drift_features = 0

for col in FEATURE_COLUMNS:
    statistic, p_value = ks_2samp(
        reference_df[col],
        current_df[col],
    )

    is_drift = p_value < DRIFT_PVALUE_THRESHOLD

    if is_drift:
        n_drift_features += 1

    drift_result[col] = {
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "drift": bool(is_drift),
    }

    drift_rows.append(
        {
            "feature": col,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift": is_drift,
        }
    )

    task.get_logger().report_scalar(
        title="KS Statistic",
        series=col,
        value=statistic,
        iteration=0,
    )

    task.get_logger().report_scalar(
        title="P Value",
        series=col,
        value=p_value,
        iteration=0,
    )

    # ------------------------------------------------
    # Histogram for drift features only
    # ------------------------------------------------

    if is_drift:
        train_hist, bins = np.histogram(
            reference_df[col],
            bins=20,
        )

        test_hist, _ = np.histogram(
            current_df[col],
            bins=bins,
        )

        task.get_logger().report_histogram(
            title=f"{col} distribution",
            series="reference",
            values=train_hist,
            iteration=0,
        )

        task.get_logger().report_histogram(
            title=f"{col} distribution",
            series="current",
            values=test_hist,
            iteration=0,
        )


# =====================================================
# Drift summary
# =====================================================

drift_ratio = n_drift_features / len(FEATURE_COLUMNS)

if drift_ratio < 0.10:
    status = "PASS"
elif drift_ratio < DRIFT_RATIO_THRESHOLD:
    status = "WARNING"
else:
    status = "FAIL"

drift_table = pd.DataFrame(drift_rows).sort_values("p_value")

max_drift_feature = drift_table.iloc[0]["feature"]
min_p_value = float(drift_table.iloc[0]["p_value"])

drift_summary = {
    "status": status,
    "n_features": len(FEATURE_COLUMNS),
    "n_drift_features": n_drift_features,
    "drift_ratio": float(drift_ratio),
    "drift_detected": bool(status == "FAIL"),
    "max_drift_feature": max_drift_feature,
    "min_p_value": min_p_value,
}

drift_lineage = {
    "current_feature_dataset_id": feature_dataset_id,
    "reference_feature_dataset_id": reference_feature_dataset_id,
    "model_id": champion_model_id,
    "train_task_id": champion_train_task_id,
    "feature_task_id": feature_task.id,
    "drift_task_id": task.id,
}

# =====================================================
# Upload artifacts
# =====================================================

task.upload_artifact(
    name="drift_result",
    artifact_object=drift_result,
)

task.upload_artifact(
    name="drift_summary",
    artifact_object=drift_summary,
)

task.upload_artifact(
    name="drift_table",
    artifact_object=drift_table,
)

task.upload_artifact(
    "drift_lineage",
    drift_lineage,
)

# =====================================================
# Log
# =====================================================

task.get_logger().report_text(
    f"""
    Drift Summary

    Status           : {status}
    Number of features      : {len(FEATURE_COLUMNS)}
    Number of drift features: {n_drift_features}
    Drift ratio      : {drift_ratio:.2f}

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
    drift_ratio,
)

task.get_logger().report_single_value(
    "n_drift_features",
    n_drift_features,
)

print(
    "Drift summary:",
    drift_summary,
)


task.close()
