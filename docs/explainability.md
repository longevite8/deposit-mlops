# Model Explainability with SHAP

## Overview

The `explain_model.py` task provides post-training model interpretability using **SHAP TreeExplainer**, optimized for LightGBM models.

This task generates:

* Feature importance rankings (mean |SHAP values|)
* SHAP dependence plots (how features affect predictions)
* Top-3 feature scalars for ClearML dashboard
* Comprehensive model behavior documentation

---

## Architecture in Training Pipeline

SHAP analysis runs **after training** in the training pipeline:

```text
Train Model
    ↓
Evaluate Model
    ↓
Register Model (candidate tag)
    ↓
Explain Model ← SHAP TreeExplainer
    ↓
Compare vs Champion
    ↓
Promote Champion
```

**Placement Rationale:**

* Runs after training produces the model
* Runs before comparison (for decision support)
* Does not block champion promotion (independent analysis)

---

## SHAP TreeExplainer

### Why TreeExplainer?

SHAP TreeExplainer is optimized for tree-based models like LightGBM:

* Polynomial time complexity (not exponential like SHAP Kernel)
* Exact SHAP values (not approximate)
* Leverages tree structure efficiently

### Input Requirements

```python
# From upstream tasks
model_id = upstream_task.artifacts["model_id"].get()
model = Model.get_model(model_id)
lgb_model = model.get_model_object()

# Feature dataset
feature_dataset_id = upstream_task.artifacts["feature_dataset_id"].get()
dataset = Dataset.get(dataset_id=feature_dataset_id)
train_df = pd.read_parquet(local_path / "train.parquet")
```

**Requirements:**

* LightGBM model object
* Training feature data (for reference background)
* Feature names

### Configuration

From `config.py`:

```python
SHAP_N_SAMPLES = 100          # Background samples for TreeExplainer
SHAP_MAX_DISPLAY = 10         # Top N features to display
```

---

## Output Artifacts

### 1. explain_summary

JSON summary of SHAP analysis:

```python
{
    "feature_importance": {
        "lag_1": 0.45,
        "rolling_mean_7": 0.35,
        "lag_7": 0.20
    },
    "top_3_features": ["lag_1", "rolling_mean_7", "lag_7"],
    "analysis_samples": 100,
    "feature_count": 3
}
```

### 2. explain_lineage

Lineage tracking:

```python
{
    "explain_task_id": "task_abc123",
    "train_task_id": "task_xyz789",
    "model_id": "model_def456",
    "feature_dataset_id": "dataset_ghi012",
    "feature_task_id": "task_jkl345",
}
```

### 3. ClearML Scalars

Dashboard-ready metrics:

```python
task.log_scalar("SHAP/lag_1_importance", 0.45)
task.log_scalar("SHAP/rolling_mean_7_importance", 0.35)
task.log_scalar("SHAP/lag_7_importance", 0.20)
```

### 4. SHAP Plots

Saved to ClearML:

* **Force plot** - Individual prediction explanation
* **Dependence plots** - Feature vs prediction relationship
* **Summary plot** - Feature importance ranking

---

## Feature Importance Interpretation

### Top Features Example

```
Feature             Mean |SHAP|
────────────────────────────────
lag_1               0.45    ← Most important
rolling_mean_7      0.35
lag_7               0.20
```

**Interpretation:**

* `lag_1` has largest average impact on predictions
* Changes in `lag_1` shift predictions ~0.45 units on average
* Model relies most on previous day's value

### SHAP Dependence Plots

Shows relationship between feature and prediction:

```
Rolling Mean (7) vs Prediction
    ^
    |     ✓ ✓
    |  ✓    ✓
    | ✓  ✓  ✓
    |─────────── Feature Value
```

**Insights:**

* Monotonic relationship → linear dependency
* Curved relationship → nonlinear interactions
* Scattered points → high variance or noise

---

## Task Signature

```python
def explain_model(
    task: Task,
    train_task_id: str,
    feature_dataset_id: str,
    model_id: str,
    n_shap_samples: int = 100
) -> None:
    """
    Explain trained model using SHAP TreeExplainer.
    
    Args:
        task: ClearML task object
        train_task_id: Upstream training task ID
        feature_dataset_id: ClearML Dataset ID with features
        model_id: Trained model ID
        n_shap_samples: Number of background samples
    """
```

---

## Two-Artifact Pattern

Always upload both artifacts:

```python
# Initialize at start
explain_lineage = {
    "explain_task_id": task.id,
    "train_task_id": train_task_id,
    "model_id": model_id,
    "feature_dataset_id": feature_dataset_id,
}

explain_summary = {
    "feature_importance": {...},
    "top_3_features": [...],
    "analysis_samples": n_shap_samples,
}

# Upload at end (even on error)
task.upload_artifact("explain_lineage", explain_lineage)
task.upload_artifact("explain_summary", explain_summary)
```

---

## Integration with Compare Champion

The `compare_champion.py` task:

* Reads `explain_summary` from explain_model task
* Can include feature importance in comparison report
* Uses feature ranking for business insights

**Example:**

```
Candidate Model (New)
  Top Feature: lag_1 (importance: 0.45)
  
Champion Model (Current)
  Top Feature: lag_1 (importance: 0.42)
  
Insight: Feature importance stable across versions
```

---

## Monitoring SHAP Output

### ClearML Dashboard

View SHAP scalars in real-time:

1. Open ClearML WebApp
2. Navigate to explain_model task
3. View "SCALARS" section:
   * `SHAP/lag_1_importance`
   * `SHAP/rolling_mean_7_importance`
   * `SHAP/lag_7_importance`

### Tracking Feature Stability

Monitor feature importance across versions:

```
Task Run 1: lag_1 = 0.45, rolling_mean_7 = 0.35
Task Run 2: lag_1 = 0.44, rolling_mean_7 = 0.36  ← Stable

Task Run 3: lag_1 = 0.20, rolling_mean_7 = 0.70  ← Drift!
```

**Action:** If feature importance shifts significantly:

* Investigate data distribution changes
* Check for feature engineering issues
* Consider retraining

---

## Common Patterns

### Pattern 1: Feature Importance Change

If top features change between runs:

* **Expected:** Minor ranking shifts (±0.02)
* **Alert:** Top-3 features completely different
* **Action:** Investigate root cause, check drift detection

### Pattern 2: Single Dominant Feature

If one feature has >>0.5 importance:

* **Check:** Is model overfitting to one feature?
* **Investigate:** Feature correlations with target
* **Consider:** Feature engineering improvements

### Pattern 3: Balanced Importance

If all features have similar importance (~0.3 each):

* **Positive:** Model uses multiple signals
* **Check:** Feature quality and coverage
* **Opportunity:** Remove low-importance features

---

## Error Handling

The task handles gracefully:

```python
# Model loading error
if model is None:
    logger.error(f"Failed to load model {model_id}")
    # Upload empty artifacts
    task.upload_artifact("explain_summary", {})
    task.upload_artifact("explain_lineage", {...})
    task.close()
    raise SystemExit(1)

# Feature data error
if len(train_df) == 0:
    logger.error("Training data is empty")
    # Upload with empty importance
    task.upload_artifact("explain_summary", {"feature_importance": {}})
    task.upload_artifact("explain_lineage", {...})
    task.close()
    raise SystemExit(1)
```

---

## Performance Considerations

### SHAP Computation Time

* **Background samples:** 100 (configurable)
* **Feature count:** 3
* **Typical runtime:** 5-30 seconds

### Memory Usage

* Store feature data in memory during SHAP computation
* Large datasets: Consider sampling

### Optimization

```python
# Use fewer background samples for speed
n_samples = 50  # Instead of 100

# Or limit to top features
feature_subset = train_df[top_10_features]
explainer = shap.TreeExplainer(lgb_model, data=feature_subset)
```

---

## Integration Checklist

* [ ] `train_model.py` uploads `model_id` artifact
* [ ] `feature_engineering.py` creates feature Dataset
* [ ] `explain_model.py` template registered in ClearML
* [ ] `TEMPLATE_EXPLAIN_ID` configured in `config.py`
* [ ] Training pipeline includes `explain_model` step
* [ ] SHAP scalars visible in ClearML dashboard
* [ ] `compare_champion.py` reads `explain_summary`
* [ ] Feature importance monitored across model versions

---

## References

* [SHAP Documentation](https://shap.readthedocs.io/)
* [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
* [ClearML Logging](https://clear.ml/docs/latest/docs/fundamentals/logging/)
