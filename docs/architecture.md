# System Architecture

## Overview

This project implements an end-to-end MLOps architecture for cashflow forecasting using ClearML with a **Champion/Challenger design pattern**.

The system manages the complete machine learning lifecycle:

* Data extraction and synthetic generation
* Feature engineering (lag and rolling features)
* Data validation (schema, dtypes, ranges, missing values)
* Drift detection (Kolmogorov-Smirnov statistical test)
* Hyperparameter optimization (Optuna)
* Model training (LightGBM with loss curves)
* Model evaluation (multi-metric with quality gates)
* Model explainability (SHAP TreeExplainer)
* Model registration and tagging (candidate/champion/archived)
* Production inference (batch predictions with latency metrics)
* Monitoring and alerting (MAPE/R²/drift thresholds)
* Continuous retraining (fire-and-forget auto-retraining)

---

## High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Champion Registry                        │
│            (Singleton ClearML Task in Registry             │
│             Project - tracks champion model ID              │
│                  & metadata)                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
    ┌──────────────┐              ┌──────────────────┐
    │   Training   │              │   Production     │
    │   Pipeline   │              │   Pipeline       │
    │   (10 steps) │              │   (7 steps)      │
    └──────────────┘              └──────────────────┘
          │                                 │
          │                                 └─────┐
          │                                       │
          └───────────────┬───────────────────────┘
                          ▼
                  (New Champion Model
                   loaded by Production)
```

---

## Main Architecture Patterns

### 1. Champion/Challenger Architecture

A **single, immutable ClearML task** in the "Cashflow Registry" project tracks the current production model:

```text
Training Pipeline                Production Pipeline
    ↓                                  ↓
Produces                         Always loads from
Candidate Models                 Champion Registry
    ↓                                  ↓
Compare vs                       Guarantees
Champion Metrics                 Deterministic
    ↓                            Inference
Promote Winner to
Champion Registry
```

**Benefits:**

* Production always uses a validated champion model
* Easy rollback by reverting champion tags
* Future Champion/Challenger A/B testing ready

### 2. Two-Artifact Lineage Pattern

Every ClearML task uploads **exactly two artifacts** for complete traceability:

* `<task>_summary` — Decision/results (JSON format)
  * Example: `compare_champion_summary = {"candidate_win": true, "candidate_model_id": "..."}`

* `<task>_lineage` — Upstream task IDs & dataset IDs
  * Example: `compare_champion_lineage = {"compare_task_id": "...", "train_task_id": "...", "feature_dataset_id": "..."}`

**Flow:**

```text
extract_task
    ↓ (upload: raw_dataset_id)
feature_task
    ↓ (upload: feature_dataset_id)
train_task
    ↓ (upload: train_summary, train_lineage)
evaluate_task
    ↓ (upload: evaluate_summary, evaluate_lineage)
compare_champion
    ↓ (upload: compare_summary, compare_lineage)
promote_champion
    └─ reads compare_lineage for full metadata
```

### 3. ClearML Dataset-Based Data Exchange

Features flow through **ClearML Dataset objects** (not direct DataFrame artifacts):

```python
# Producer (feature_engineering.py)
dataset = Dataset.create(dataset_project="Datasets")
dataset.add_files(tmp_dir)  # train.parquet, valid.parquet, test.parquet
dataset.upload()
dataset.finalize()
task.upload_artifact("feature_dataset_id", dataset.id)

# Consumers (train, hpo, validate, drift, inference, etc.)
feature_dataset_id = upstream_task.artifacts["feature_dataset_id"].get()
dataset = Dataset.get(dataset_id=feature_dataset_id)
local_path = Path(dataset.get_local_copy())
train_df = pd.read_parquet(local_path / "train.parquet")
```

**Advantages:**

* Version tracking of feature datasets
* Reproducible downstream tasks
* Clear lineage in ClearML UI

### 4. Template/Parameter Guard Pattern

All task scripts are **ClearML task templates**. When parameters are empty, tasks exit cleanly:

```python
task = Task.init(project_name=PROJECT_TEMPLATE, task_name=TEMPLATE_NAME, task_type="training")
params = task.connect({"upstream_task_id": ""})

if not params["upstream_task_id"]:  # Template mode guard
    task.close()
    raise SystemExit(0)

# Normal execution continues...
```

**Pipeline instantiation** clones templates with parameter overrides:

```python
pipeline.add_step(base_task_id=template_id, parameter_override={"upstream_task_id": upstream_id})
```

### 5. Model Tagging via APIClient

Model tag mutations **must use APIClient**, not the Model property setter:

```python
from clearml.backend_api.session.client import APIClient

client = APIClient()
client.models.edit(model=model_id, tags=["candidate", "v1.0"])
```

Used in: `register_model.py`, `promote_champion.py`

### 6. Fire-and-Forget Auto-Retraining

Auto-retraining trigger uses **immediate task.close()** to prevent Agent deadlock:

```python
# In alerting_model.py
need_retraining = (mape > THRESHOLD) or (r2 < THRESHOLD) or (drift == "FAIL")
# ... generate alert ...

# In auto_retraining.py (triggered by alert_task_id)
pipeline = PipelineController(...)
pipeline.add_step(...)
pipeline.start()
task.close()  # ← CRITICAL: Free agent immediately
raise SystemExit(0)  # Do NOT wait for child pipeline
```

**Why this pattern?**

* ClearML Agent is single-threaded
* Parent task waiting for child → child has no agent to run
* Immediate close() releases slot for child pipeline

### 7. Metadata Key Convention

Model metadata stored with **exact keys** (enforced across all code):

```python
{
    "train_task_id": "...",          # NOT "training_task_id"
    "feature_task_id": "...",
    "feature_dataset_id": "...",     # ClearML Dataset ID
    "raw_dataset_id": "...",
    "mape": 0.15,
    "r2": 0.92,
    "hpo_task_id": "...",
    "evaluate_task_id": "...",
    "register_task_id": "...",
    "compare_task_id": "..."
}
```

---

## Component Subsystems

### Training Pipeline (10 Steps)

```text
Extract → Feature → Validate → Drift → HPO → Train → Evaluate → Register → Explain → Compare → Promote
```

**Responsibilities:**
* Generate synthetic/real data
* Create features (lag_1, lag_7, rolling_mean_7)
* Validate data quality
* Check for data drift vs champion reference
* Optimize hyperparameters (Optuna)
* Train candidate model (LightGBM)
* Evaluate on test set with quality gates
* Register model with "candidate" tag
* Analyze model via SHAP
* Compare candidate vs champion metrics
* Promote candidate to champion (if winning)

**Caching Strategy:**
* Deterministic steps (extract, feature, validate, drift, hpo): **Cached** by ClearML
* Model-dependent steps (train, evaluate, register, compare, promote): **No cache**

See [training_pipeline.md](training_pipeline.md) for details.

### Production Pipeline (7 Steps)

```text
Extract → Feature → Drift → Inference → Monitoring → Alerting → Auto-Retraining
```

**Responsibilities:**
* Load fresh production data
* Apply feature engineering (identical to training)
* Detect data drift vs champion reference
* Run batch inference with champion model
* Track MAPE/R² metrics and drift ratio
* Generate email alerts if thresholds exceeded
* Trigger training pipeline if alert generated

**Caching Strategy:**
* Deterministic steps (extract, feature): **Cached**
* Live monitoring steps (drift, inference, monitoring, alerting, auto_retraining): **No cache**

See [production_pipeline.md](production_pipeline.md) for details.

### Champion Registry (Singleton)

A **single, immutable ClearML task** that tracks the champion model:

```python
{
    "champion_model_id": "model_abc123",
    "champion_train_task_id": "task_xyz789",
    "champion_feature_dataset_id": "dataset_def456",
    "champion_feature_task_id": "task_ghi012",
    "mape": 0.15,
    "r2": 0.92,
    "promoted_at": "2024-07-02 15:30:00"
}
```

**Location:**
* Project: `"Deposit-CashFlow/Registry"` (singleton)
* Never queried directly; only loaded by production pipeline
* Updated via `promote_champion.py`

**Benefits:**
* Single source of truth for production model
* Deterministic inference (no race conditions)
* Version history in ClearML task log

See [champion_registry.md](champion_registry.md) for details.

### Drift Detection

Detects data distribution shifts using **Kolmogorov-Smirnov (KS) test**:

```
Production Data vs Champion Training Reference Data
        ↓
    KS test per column
        ↓
    Status: PASS / WARNING / FAIL
```

**Thresholds:**
* `DRIFT_PVALUE_THRESHOLD = 0.05` (p-value cutoff)
* `DRIFT_RATIO_THRESHOLD = 0.2` (max % of drifted columns)

**First Run Behavior:**
* No champion reference → uses current data as reference
* Result: Always `PASS` (graceful degradation)

See [monitoring_and_drift.md](monitoring_and_drift.md) for details.

### Monitoring & Alerting

Real-time production system health tracking:

```
Inference Results + Drift Status
        ↓
    Monitoring Task
        ↓
    Check Thresholds
        ↓
    Generate Alert Summary
        ↓
    Email Notification
        ↓
    Trigger Auto-Retraining (if needed)
```

**Alert Triggers:**
* MAPE > `MONITORING_MAPE_THRESHOLD` (0.20)
* R² < `MONITORING_R2_THRESHOLD` (0.85)
* Drift Status = `"FAIL"`

**Email Setup:**
* Via Gmail App Password in `.env`
* Configured in `config.py`
* Uses STARTTLS encryption

See [monitoring_and_drift.md](monitoring_and_drift.md) for details.

### Model Explainability (SHAP)

Post-training SHAP TreeExplainer analysis:

```
Champion Model + Training Features
        ↓
    SHAP TreeExplainer
        ↓
    Feature Importance Rankings
    SHAP Dependence Plots
    Top-3 Feature Scalars
        ↓
    Uploaded to ClearML
```

**Features:**
* TreeExplainer optimized for LightGBM
* Feature importance (mean |SHAP|)
* Dependence plots for top features
* Scalars for ClearML dashboard

See [explainability.md](docs/explainability.md) for details.

---

## Data Flow Architecture

### Training Data Flow

```
Extract Data
    ↓
Create Features (lag_1, lag_7, rolling_mean_7)
    ↓
ClearML Dataset
(train.parquet, valid.parquet, test.parquet)
    ↓
Validation Checks
    ↓
Drift Detection (vs reference or current)
    ↓
HPO (Optuna) + Training (LightGBM)
    ↓
Evaluation on Test Set
    ↓
Model Registry (tagged "candidate")
    ↓
SHAP Explainability Analysis
    ↓
Champion Comparison
    ↓
Champion Promotion (if winning)
```

### Production Data Flow

```
Load Champion Model
    ↓
Extract Fresh Data
    ↓
Apply Feature Engineering
    ↓
Drift Detection (vs champion reference)
    ↓
Batch Inference (Predictions)
    ↓
Monitor MAPE/R²/Drift
    ↓
Generate Alerts (if thresholds exceeded)
    ↓
Auto-Retraining Trigger (if alert)
    └─ Clones Training Pipeline
    └─ Enqueues to SERVICES_QUEUE
    └─ Returns immediately (fire-and-forget)
```

---

## Project Hierarchy

ClearML project structure (hierarchical with `/` separator):

```
Deposit-CashFlow
├── /Templates              # 15 task template definitions
├── /Pipelines              # Training and production pipelines
├── /Datasets               # Feature datasets (parquet)
└── /Registry               # Champion registry singleton
```

All tasks created with `Task.init(project_name=PROJECT_NAME, ...)` use the hierarchical structure.

---

## Common Architecture Pitfalls

| Pitfall | Fix |
|---|---|
| Reading DataFrame directly from task artifacts | Read from ClearML Dataset via `feature_dataset_id` |
| Using `model.tags +=` instead of APIClient | Always use `APIClient().models.edit(model=..., tags=...)` |
| Forgetting `compare_champion_lineage` upload | Upload lineage in EVERY exit path (success, no champion, invalid) |
| Parent agent waiting for child pipeline | Use fire-and-forget pattern: `task.close()` immediately after trigger |
| Wrong metadata key `"training_task_id"` | Always use `"train_task_id"` |
| Drift crash when no champion exists | Handle gracefully: use current data as reference, return PASS |
| Feature engineering inconsistencies | Keep canonical function in `src/features.py`, import in all tasks |
| Caching enabled on non-deterministic steps | Cache only: extract, feature, validate, drift, hpo (deterministic) |

---

## Next Steps

1. **[training_pipeline.md](training_pipeline.md)** - Understand 10-step training workflow
2. **[production_pipeline.md](production_pipeline.md)** - Understand 7-step production workflow
3. **[explainability.md](explainability.md)** - Learn SHAP analysis details
4. **[agent_setup.md](agent_setup.md)** - Configure ClearML Agent for distributed execution

* Batch inference
* Drift monitoring
* Prediction monitoring
* Alerting

Documentation:

* docs/production_pipeline.md

---

## Monitoring System

Responsible for:

* Data drift monitoring
* Prediction monitoring
* Alerting
* Retraining triggers

Documentation:

* docs/monitoring_and_drift.md

---

# Training Pipeline

```text
Extract
↓
Feature
↓
Validate
↓
Drift
↓
HPO
↓
Train
↓
Evaluate
↓
Register
↓
Update Champion Registry
```

The training pipeline produces a candidate model.

If the model passes quality gates, the Champion Registry is updated.

---

# Production Pipeline

```text
                 Load Champion Registry
                          │
                          ▼

Extract
↓
Feature
├──────── Drift ─────────┐
│                        │
└────── Inference ───────┘
             │
             ▼
         Monitoring
             │
             ▼
          Alerting
```

Production inference always uses the current champion model.

---

# Artifact Flow

## Training Pipeline

```text
Extract
    ↓
raw_df

Feature
    ↓
train_df
valid_df
test_df

Validate
    ↓
validation_report

Drift
    ↓
drift_report

HPO
    ↓
best_params

Train
    ↓
model
model_id

Evaluate
    ↓
evaluation_result

Register
    ↓
candidate model

Update Champion Registry
    ↓
champion_registry
```

---

## Production Pipeline

```text
Load Champion Registry
    ↓
champion_registry

Extract
    ↓
raw_df

Feature
    ↓
feature_df

Drift
    ↓
drift_report

Inference
    ↓
prediction_df

Monitoring
    ↓
monitoring_report

Alerting
    ↓
alerts
```

---

# Design Principles

The architecture follows several principles.

## Reproducibility

Every task records:

* Git commit
* Parameters
* Artifacts
* Metrics

This enables deterministic reruns.

---

## Loose Coupling

Tasks communicate through artifacts rather than direct code dependencies.

This minimizes cascading changes.

---

## Single Responsibility

Each task performs one job only.

Examples:

* Validate task only validates data.
* Drift task only detects drift.
* Register task only publishes models.

---

## Minimal Artifact Duplication

Datasets are not copied between tasks unnecessarily.

For example:

* Drift only produces drift_report.
* HPO loads datasets from Validate task.
* Monitoring consumes outputs from Inference and Drift.

---

## Champion Registry Pattern

The production system never searches for the latest model.

Instead, it reads the current champion model from a singleton Champion Registry.

This avoids:

* Model.query_models()
* Task.get_tasks()
* "Latest task" assumptions

and guarantees deterministic inference.

---

# Future Roadmap

The architecture is designed to support:

* Human-in-the-loop retraining
* Automatic retraining
* Champion / Challenger
* Rollback
* Model versioning
* Canary deployment
* A/B testing
* Online serving
* Feature store
* CI/CD integration

without requiring major refactoring.
