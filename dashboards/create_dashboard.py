"""
Tạo ClearML Dashboard để monitor training + production metrics.
Chạy một lần: python dashboards/create_dashboard.py
"""

from clearml import Task

# =====================================================
# Dashboard: Training Metrics
# =====================================================

training_dashboard = Task.init(
    project_name="Dashboards",
    task_name="Training Metrics Dashboard",
    task_type=Task.TaskTypes.monitor,
)

training_dashboard.add_function_step(
    name="HPO Metrics",
    function=None,  # Just a marker
    function_kwargs={},
    function_return=[],
)

# Link tasks để hiển thị metrics
training_dashboard.get_logger().report_text(
    markdown="""
# Training Pipeline Metrics

## HPO Performance
- **Best MAPE**: Từ HPO task
- **Number of Trials**: Từ HPO task
- **Trial Convergence**: Plot từ HPO task

## Model Training
- **Training Loss**: Từ Training task (loss curves)
- **Validation Loss**: Từ Training task
- **Feature Importance**: Top 5 features từ Training task

## Model Evaluation
- **Test MAPE**: Từ Evaluation task
- **Test R2**: Từ Evaluation task
- **Quality Gate Status**: Passed/Failed

## Model Explainability
- **SHAP Feature Importance**: Từ Explain task
- **Top 5 Contributing Features**: Từ Explain task
""",
)

# =====================================================
# Dashboard: Production Metrics
# =====================================================

production_dashboard = Task.init(
    project_name="Dashboards",
    task_name="Production Metrics Dashboard",
    task_type=Task.TaskTypes.monitor,
)

production_dashboard.get_logger().report_text(
    markdown="""
# Production Pipeline Metrics

## Inference Performance
- **Inference Latency (ms/sample)**: Từ Inference task
- **Batch Size**: Từ Inference task
- **Total Inference Time**: Từ Inference task

## Drift Detection
- **Drift Status**: PASS / WARNING / FAIL
- **KS Test p-value**: Từ Drift Detection task
- **Ratio of Drifted Features**: Từ Drift Detection task

## Model Monitoring
- **Current MAPE (Production)**: Từ Monitoring task
- **Current R2 (Production)**: Từ Monitoring task
- **Need Retraining**: Yes/No decision

## Alerting
- **Alert Status**: ALERT / OK
- **Email Sent**: Yes/No
- **Last Alert Time**: Timestamp
- **Reason**: Text description
""",
)

# =====================================================
# Dashboard: Model Comparison
# =====================================================

comparison_dashboard = Task.init(
    project_name="Dashboards",
    task_name="Champion vs Candidate Dashboard",
    task_type=Task.TaskTypes.monitor,
)

comparison_dashboard.get_logger().report_text(
    markdown="""
# Champion vs Candidate Comparison

## Metrics Comparison
| Metric | Champion | Candidate | Winner |
|--------|----------|-----------|--------|
| MAPE | From Compare task | From Compare task | ✓ |
| R2 | From Compare task | From Compare task | ✓ |
| Training Time | From Model metadata | From Compare task | ✓ |

## Promotion Status
- **Candidate Won**: Yes/No
- **Reason**: Delta metrics comparison
- **New Champion ID**: If promoted
- **Previous Champion ID**: If promoted
""",
)

print("✅ Dashboards created successfully!")
print("   - Training Metrics Dashboard")
print("   - Production Metrics Dashboard")
print("   - Champion vs Candidate Dashboard")
print("\n📊 View them in ClearML Web UI → Dashboards")

training_dashboard.close()
production_dashboard.close()
comparison_dashboard.close()
