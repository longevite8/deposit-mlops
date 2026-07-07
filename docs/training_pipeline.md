# Training Pipeline

## Overview

The training pipeline is responsible for producing a candidate model.

The pipeline performs:

* Data extraction
* Feature engineering
* Data validation
* Drift detection
* Hyperparameter optimization
* Model training
* Model evaluation
* Model registration
* Updating the Champion Registry

---

# Pipeline DAG

```text
Extract
    ↓
Feature
    ↓
Validate → Drift → HPO → Train → Evaluate → Register → Explain → Compare → Promote
```

**Caching Strategy:**

* **Cached (Deterministic):** Extract, Feature, Validate, Drift, HPO
* **Not Cached (Model-Dependent):** Train, Evaluate, Register, Explain, Compare, Promote

**New Step:** `Explain` (SHAP analysis) added after `Evaluate`, before `Compare`

---

# Task Descriptions

## 1. Extract

### Purpose

Load raw data from source systems.

### Outputs

Artifact:

* raw_df

---

## 2. Feature

### Purpose

Create machine learning features.

### Inputs

Artifact:

* raw_df

### Outputs

Artifacts:

* train_df
* valid_df
* test_df

---

## 3. Validate

### Purpose

Verify data quality.

### Checks

* Schema validation
* Missing values
* Dtype validation
* Range validation

### Inputs

Artifacts:

* train_df
* valid_df
* test_df

### Outputs

Artifact:

* validation_report

The datasets themselves remain stored in Feature task.

---

## 4. Drift

### Purpose

Detect data drift before training.

### Inputs

Datasets are loaded from:

* Feature task

Validation result is loaded from:

* Validate task

### Outputs

Artifact:

* drift_report

Drift task does not duplicate datasets.

---

## 5. HPO

### Purpose

Search for the best hyperparameters.

### Inputs

Datasets are loaded from:

* Feature task

Drift information is loaded from:

* Drift task

### Outputs

Artifact:

* best_params

---

## 6. Train

### Purpose

Train the final model.

### Inputs

Datasets:

* Feature task

Hyperparameters:

* HPO task

### Outputs

Artifacts:

* model
* model_id

---

## 7. Evaluate

### Purpose

Evaluate model quality.

### Inputs

Datasets:

* Feature task

Model:

* Train task

### Metrics

* MAPE
* R²

### Outputs

Artifact:

* evaluation_result

---

## 9. Explain

### Purpose

Generate SHAP TreeExplainer analysis for model interpretability.

### Inputs

Model:

* Train task

Features:

* Feature task

### Outputs

Artifacts:

* explain_summary (feature importance rankings)
* explain_lineage (upstream task tracking)

ClearML Scalars:

* SHAP/feature_name_importance (top-3 features)

---

## 10. Compare

### Purpose

Compare candidate model against champion model.

### Inputs

Candidate model:

* Register task

Champion model:

* Champion Registry (if exists)

Explanations:

* Explain task (SHAP analysis)

### Outputs

Artifacts:

* compare_summary {"candidate_win": bool, "candidate_model_id": "..."}
* compare_lineage (full upstream tracking)

### Logic

```python
if no champion exists:
    candidate_win = True  # First model always wins

if candidate MAPE < champion MAPE:
    candidate_win = True
elif candidate MAPE == champion MAPE:
    if candidate R² > champion R²:
        candidate_win = True
    else:
        candidate_win = False
else:
    candidate_win = False
```

---

## 11. Promote

### Purpose

Update Champion Registry if candidate wins.

### Inputs

Comparison result:

* Compare task (compare_summary, compare_lineage)

Champion model tags:

* Swap "champion" → "archived"
* Swap "candidate" → "champion"

### Outputs

Artifacts:

* promote_summary {"promoted": bool}
* promote_lineage (full tracking)

### Tag Changes

```
Before (if promoting):
  Old Champion: tags = ["champion"]
  Candidate:    tags = ["candidate"]

After:
  Old Champion: tags = ["archived"]
  Candidate:    tags = ["champion"]  ← New Champion Registry points here
```

---

# Artifact Flow

```text
Extract
    ↓
raw_df (ClearML Dataset)

Feature
    ↓
feature_dataset_id (train/valid/test parquet)

Validate
    ↓
validation_report

Drift
    ↓
drift_report
drift_lineage

HPO
    ↓
hpo_summary (best_params)
hpo_lineage

Train
    ↓
model_id
train_summary
train_lineage

Evaluate
    ↓
evaluate_summary (metrics)
evaluate_lineage

Register
    ↓
model (tagged "candidate")
register_summary
register_lineage

Explain
    ↓
explain_summary (feature importance)
explain_lineage

Compare
    ↓
compare_summary (candidate_win: bool)
compare_lineage

Promote
    ↓
Champion Registry updated (if winning)
promote_summary
promote_lineage
```

---

# Design Principles

## Two-Artifact Lineage Pattern

Every task uploads:

* `<task>_summary` — Decision/result
* `<task>_lineage` — Upstream tracking

Ensures complete traceability for:

* Debugging
* Auditing
* Reproduction

---

## No Dataset Duplication

ClearML Datasets stored once:

* Feature task creates: `train.parquet`, `valid.parquet`, `test.parquet`
* All downstream tasks load via `dataset.get_local_copy()`
* Version tracked by ClearML Dataset ID

---

## Model Tagging (APIClient)

Always use `APIClient` for tag mutations:

```python
from clearml.backend_api.session.client import APIClient
client = APIClient()
client.models.edit(model=model_id, tags=["candidate"])  # NOT model.tags +=
```

Prevents race conditions and ensures correctness.

---

## Champion Registry (Singleton)

Single immutable task tracking champion:

* Located in `"Deposit-CashFlow/Registry"` project
* Loaded by production pipeline for deterministic inference
* Updated only via `promote_champion.py`
* Enables rollback and future A/B testing

---

## Deterministic Training

Reproducible because:

* Parameters tracked via ClearML task.connect()
* Features versioned via ClearML Dataset
* Git commit recorded
* All metrics logged
* Artifacts timestamped

Rerun same task ID → Get same results.

---

# Common Patterns

## Early Exit (Template/Parameter Guard)

All tasks use this pattern:

```python
task = Task.init(project_name=PROJECT_TEMPLATE, task_name=TEMPLATE_NAME)
params = task.connect({"upstream_task_id": ""})

if not params["upstream_task_id"]:  # Template mode
    task.close()
    raise SystemExit(0)

# Normal execution...
```

When registered as ClearML task templates, parameters are empty → early exit.

When instantiated via pipeline, parameters provided → normal execution.

---

## Cache for Deterministic Steps

Configured in `pipelines/trainning_pipeline.py`:

```python
pipeline.add_step(
    name="extract",
    base_task_id=TEMPLATE_EXTRACT_ID,
    ...
    cache_base_task_id=TEMPLATE_EXTRACT_ID  # Enable caching
)
```

ClearML caches output when parameters unchanged.

Benefits:

* Skips re-extraction if data unchanged
* Saves time on iterative development
* Disabled for model-dependent steps (train, evaluate, register, compare, promote)

---

## Quality Gates

Each task can enforce quality thresholds:

```python
params = task.connect({
    "MAPE_THRESHOLD": 0.25,
    "R2_THRESHOLD": 0.80,
})

if mape > params["MAPE_THRESHOLD"]:
    logger.warning(f"MAPE {mape} exceeds threshold {params['MAPE_THRESHOLD']}")
    # Don't publish if quality gate fails
```

Configurable per task run without code change.

---

# Future Extensions

This architecture supports:

* Human-in-the-loop model approval
* Automatic retraining (implemented in production pipeline)
* Champion/Challenger A/B testing (registry pattern ready)
* Rollback (archived models retained)
* Model versioning (ClearML version control)
* Canary deployment (progressive traffic shift)

without major architectural changes.
