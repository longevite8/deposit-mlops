# Production Pipeline

## Overview

The production pipeline is responsible for:

* Loading the current champion model
* Creating features from fresh data
* Detecting drift
* Running inference
* Monitoring prediction quality
* Triggering alerts

Unlike the training pipeline, the production pipeline does not train models.

It always uses the current champion model.

---

# Pipeline DAG

```text id="bvv0ui"
Champion Registry (Singleton)
             │
             ▼
Load Champion Registry

Extract
↓
Feature
├──────── Drift ─────────┐
│                        │
└────── Inference ───────┘
          ▲
          │
Load Champion Registry

             ↓
         Monitoring
             ↓
          Alerting
```

---

# Task Descriptions

## 1. Load Champion Registry

### Purpose

Load metadata of the current production model.

### Parent Tasks

None.

This task is completely independent.

### Source

Champion Registry singleton task.

### Outputs

Artifact:

* champion_registry

Example:

```python id="jwjgyf"
{
    "version": 12,
    "champion_model_id": "...",
    "champion_train_task_id": "...",
    "mape": ...,
    "r2": ...
}
```

---

## 2. Extract

### Purpose

Load the latest production data.

### Outputs

Artifact:

* raw_df

---

## 3. Feature

### Purpose

Generate features for inference.

### Inputs

Artifact:

* raw_df

### Outputs

Artifact:

* feature_df

---

## 4. Drift

### Purpose

Monitor feature drift.

### Inputs

Features:

* feature_df

Reference distributions:

* training artifacts

### Outputs

Artifact:

* drift_report

Example:

```python id="nmmkrv"
{
    "drift_detected": False,
    "psi_max": 0.08
}
```

---

## 5. Inference

### Purpose

Generate predictions using the champion model.

### Parent Tasks

* Feature
* Load Champion Registry

### Inputs

Features:

* feature_df

Champion model metadata:

* champion_registry

### Outputs

Artifacts:

* prediction_df
* prediction_summary

---

## 6. Monitoring

### Purpose

Aggregate prediction statistics and drift statistics.

### Parent Tasks

* Inference
* Drift

### Inputs

Prediction outputs:

* prediction_df

Drift outputs:

* drift_report

### Outputs

Artifact:

* monitoring_report

Example:

```python id="yzxjns"
{
    "n_predictions": 1000,
    "prediction_mean": 105.2,
    "drift_detected": False,
    "psi_max": 0.08
}
```

---

## 7. Alerting

### Purpose

Generate warnings when abnormal conditions occur.

### Parent Tasks

* Monitoring

### Inputs

Artifact:

* monitoring_report

### Possible Alerts

* Drift detected
* Prediction anomaly
* Missing predictions
* Data quality problems

### Outputs

Artifact:

* alert_report

Example:

```python id="vftwvt"
{
    "need_retraining": False,
    "severity": "normal"
}
```

---

# Artifact Flow

```text id="7vth8e"
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
alert_report
```

---

# Dependency Graph

```text id="rwg83s"
Load Champion Registry
          │
          ▼

Extract
↓
Feature
├──────── Drift ─────────┐
│                        │
└────── Inference ───────┘
          ▲
          │
Load Champion Registry

             ↓
         Monitoring
             ↓
          Alerting
```

---

# Design Principles

## Independent Champion Model Loading

Inference never searches for the latest model.

It only consumes:

* champion_registry

This guarantees deterministic inference.

---

## Monitoring Depends on Drift

Monitoring combines:

* prediction statistics
* drift statistics

to create a unified monitoring report.

---

## Loose Coupling

Tasks communicate through artifacts.

No task directly depends on another task's code.

---

## Single Responsibility

Each task has a single responsibility.

Examples:

* Drift only detects drift.
* Inference only predicts.
* Monitoring only aggregates metrics.
* Alerting only creates alerts.

---

# Daily Execution

Typical workflow:

```text id="m3s6hz"
Day T

New data arrives
↓
Production pipeline runs
↓
Predictions generated
↓
Monitoring report created
↓
Alert generated (if needed)
↓
System keeps using current champion model
```

---

# Future Extensions

This architecture is designed to support:

* Human-in-the-loop retraining
* Automatic retraining
* Champion / Challenger
* Canary deployment
* A/B testing
* Online serving

without major refactoring.
