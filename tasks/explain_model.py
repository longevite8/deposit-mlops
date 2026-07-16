"""
Model Explainability Task — Sử dụng SHAP để phân tích đóng góp của features.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from clearml import (
    Model,
    Task,
    InputModel,
    Dataset,
)

from config import (
    PROJECT_TEMPLATE,
    TEMPLATE_EXPLAIN_NAME,
    FEATURE_COLUMNS,
    N_SHAP_SAMPLES,
)

from helpers import wait_for_artifact

task = Task.init(
    project_name=PROJECT_TEMPLATE,
    task_name=TEMPLATE_EXPLAIN_NAME,
    task_type=Task.TaskTypes.qc,
)


# =====================================================
# Parameters
# =====================================================

params = task.connect(
    {
        "feature_task_id": "",
        "train_task_id": "",
        "n_samples": N_SHAP_SAMPLES,  # Số lượng mẫu dùng cho SHAP (để tính nhanh)
    }
)


# =====================================================
# Template mode guard
# =====================================================

if not params["feature_task_id"] or not params["train_task_id"]:
    task.get_logger().report_text("Template creation mode.")
    task.close()
    raise SystemExit(0)


# =====================================================
# Load trained model identity first so non-tree models can skip SHAP safely
# =====================================================

train_task = Task.get_task(task_id=params["train_task_id"])
model_id = wait_for_artifact(
    train_task,
    "model_id",
    max_retries=10,
    wait_interval=2.0,
    logger_obj=task,
)
registered_model = Model(model_id=model_id)

if registered_model.get_metadata("model_framework") == "NeuralForecast":
    feature_task = Task.get_task(task_id=params["feature_task_id"])
    feature_dataset_id = wait_for_artifact(
        feature_task,
        "feature_dataset_id",
        max_retries=10,
        wait_interval=2.0,
        logger_obj=task,
    )
    explain_summary = {
        "model_id": model_id,
        "model_framework": "NeuralForecast",
        "status": "skipped",
        "reason": "SHAP TreeExplainer is only supported for the legacy LightGBM model.",
    }
    explain_lineage = {
        "explain_task_id": task.id,
        "train_task_id": params["train_task_id"],
        "feature_task_id": params["feature_task_id"],
        "model_id": model_id,
        "feature_dataset_id": feature_dataset_id,
    }
    task.upload_artifact("explain_summary", explain_summary)
    task.upload_artifact("explain_lineage", explain_lineage)
    task.get_logger().report_text(
        "Skipping SHAP explainability because the trained model is NeuralForecast."
    )
    print(explain_summary)
    task.flush()
    task.close()
    raise SystemExit(0)


# =====================================================
# Load feature dataset
# =====================================================

task.get_logger().report_text("Loading feature dataset...")

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

# Load training data để lấy mẫu cho SHAP
train_df = pd.read_parquet(local_path / "train.parquet")
X_train = train_df[FEATURE_COLUMNS]


# =====================================================
# Load trained model
# =====================================================

task.get_logger().report_text("Loading trained model...")

input_model = InputModel(model_id=model_id)

import joblib
import shap

model_path = input_model.get_local_copy()
model = joblib.load(model_path)


# =====================================================
# Initialize SHAP Explainer
# =====================================================

task.get_logger().report_text(
    "Initializing SHAP TreeExplainer (dùng TreeExplainer cho LightGBM)..."
)

explainer = shap.TreeExplainer(model)

# Lấy mẫu dữ liệu để tính nhanh (SHAP tính toán đắt đỏ)
n_samples = min(int(params["n_samples"]), len(X_train))
X_sample = X_train.sample(n=n_samples, random_state=42)

task.get_logger().report_text(f"Computing SHAP values for {n_samples} samples...")

shap_values = explainer.shap_values(X_sample)


# =====================================================
# Calculate feature importance (mean |SHAP|)
# =====================================================

task.get_logger().report_text("Calculating feature importance...")

# SHAP values là array của shape (n_samples, n_features)
# Lấy giá trị tuyệt đối trung bình theo từng feature
if isinstance(shap_values, list):  # Multi-class case
    mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

feature_importance_shap = pd.DataFrame(
    {
        "feature": FEATURE_COLUMNS,
        "shap_importance": mean_abs_shap,
    }
).sort_values("shap_importance", ascending=False)


# =====================================================
# Report feature importance table
# =====================================================

task.get_logger().report_table(
    title="SHAP Feature Importance",
    series="shap",
    iteration=0,
    table_plot=feature_importance_shap,
)

task.get_logger().report_text(
    f"Top 3 features by SHAP importance:\n{feature_importance_shap.head(3).to_string()}"
)


# =====================================================
# Generate dependence plots cho top 5 features
# =====================================================

task.get_logger().report_text("Generating SHAP dependence plots...")

top_5_features = feature_importance_shap.head(5)["feature"].tolist()

for feature in top_5_features:
    feature_idx = FEATURE_COLUMNS.index(feature)

    # Tạo scatter plot: feature value vs SHAP value
    if isinstance(shap_values, list):
        shap_vals_feature = shap_values[0][:, feature_idx]
    else:
        shap_vals_feature = shap_values[:, feature_idx]

    scatter_data = list(
        zip(
            X_sample[feature].values.tolist(),
            shap_vals_feature.tolist(),
        )
    )

    task.get_logger().report_scatter2d(
        title=f"SHAP Dependence: {feature}",
        series=feature,
        iteration=0,
        scatter=scatter_data,
    )


# =====================================================
# Upload artifacts
# =====================================================

explain_summary = {
    "model_id": model_id,
    "n_samples": n_samples,
    "top_features": top_5_features,
    "mean_abs_shap": {f: float(v) for f, v in zip(FEATURE_COLUMNS, mean_abs_shap)},
}

task.upload_artifact("explain_summary", explain_summary)

explain_lineage = {
    "explain_task_id": task.id,
    "train_task_id": params["train_task_id"],
    "feature_task_id": params["feature_task_id"],
    "model_id": model_id,
    "feature_dataset_id": feature_dataset_id,
}

task.upload_artifact("explain_lineage", explain_lineage)


# =====================================================
# Report scalars
# =====================================================

# Log top 10 features as scalars
# Sử dụng enumerate để đếm rank chính xác từ 1 đến 10
for rank, (_, row) in enumerate(feature_importance_shap.head(10).iterrows(), start=1):
    feature_name = row["feature"]
    importance = float(row["shap_importance"])

    # Gắn trực tiếp tên feature vào key để hiển thị rõ ràng trên UI
    # Tránh truyền string vào value của report_single_value
    task.get_logger().report_single_value(
        name=f"Top_{rank}_{feature_name}",
        value=importance,
    )

# =====================================================
# Generate markdown report
# =====================================================

markdown_report = f"""
# Model Explainability Report (SHAP)

## Summary
- **Model ID**: {model_id}
- **Samples Analyzed**: {n_samples}
- **Total Features**: {len(FEATURE_COLUMNS)}

## Top 5 Contributing Features (by SHAP importance)

| Rank | Feature | SHAP Importance |
|------|---------|-----------------|
"""

for i, row in feature_importance_shap.head(5).iterrows():
    markdown_report += (
        f"| {i + 1} | {row['feature']} | {row['shap_importance']:.6f} |\n"
    )

markdown_report += """
## Interpretation

- **SHAP Importance**: Mean absolute SHAP value cho mỗi feature
  - Giá trị cao = Feature có ảnh hưởng lớn đến predictions
  - Giá trị thấp = Feature có ảnh hưởng nhỏ đến predictions

- **Dependence Plots**: Hiển thị mối quan hệ giữa feature value và SHAP value
  - Giúp hiểu liệu feature tăng hay giảm predictions

## Next Steps

1. Review top features trong production model
2. Nếu top features thay đổi → có thể cần retraining
3. Nếu feature không quan trọng → xem xét loại bỏ để giảm complexity
"""

task.get_logger().report_text(markdown_report)


# =====================================================
# Final summary
# =====================================================

task.get_logger().report_text(
    "✅ Model explainability analysis complete!\n"
    "   - Artifacts: explain_summary, explain_lineage\n"
    "   - Scalars: 10 shap_importance values\n"
    "   - Plots: 5 dependence plots\n"
    "   - Report: Markdown summary"
)

print(explain_summary)

task.flush()
task.close()
