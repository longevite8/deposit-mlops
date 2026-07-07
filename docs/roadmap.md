# Roadmap

## Overview

This document describes the evolution roadmap of the cashflow forecasting MLOps platform.

The project is implemented incrementally to keep the codebase maintainable and production-ready.

---

# Current Architecture

The system currently supports:

* Training pipeline
* Production pipeline
* Data validation
* Drift detection
* Monitoring
* Alerting
* Champion Registry

---

# Phase 1

## Basic Training Pipeline

### Goals

Build a reproducible training workflow.

---

### Components

```text
Extract
↓
Feature
↓
HPO
↓
Train
↓
Evaluate
↓
Register
```

---

### Deliverables

* Feature engineering
* Hyperparameter optimization
* Model training
* Evaluation
* Model registration

---

# Phase 2

## Data-Centric MLOps

Focus on data quality and observability.

---

# Phase 2.3.1

## Data Validation

Pipeline:

```text
Feature
↓
Validate
```

Checks:

* Schema validation
* Missing values
* Dtype validation
* Range validation

---

# Phase 2.3.2

## Drift Detection

Pipeline:

```text
Validate
↓
Drift
↓
HPO
```

Deliverables:

* drift_report
* PSI statistics

---

# Phase 2.3.2.2

## Drift Report

Visualize drift metrics.

Deliverables:

* Markdown summary
* PSI charts
* Drift artifacts

---

# Phase 2.3.3

## Monitoring

Pipeline:

```text
Inference
+
Drift
↓
Monitoring
```

Deliverables:

* prediction statistics
* monitoring_report

---

# Phase 2.3.4

## Auto Retraining

Future phase.

Workflow:

```text
Drift detected
↓
Training pipeline
↓
Register
```

Status:

Planned.

---

# Phase 2.4

## Champion / Challenger

Future phase.

Workflow:

```text
Champion
│
└── Challenger
```

Status:

Planned.

---

# Phase 3

## Production System

Focus on inference and observability.

---

# Phase 3.1

## Production Inference

Pipeline:

```text
Extract
↓
Feature
↓
Inference
```

Deliverables:

* prediction_df
* prediction_summary

Status:

Completed.

---

# Phase 3.2

## Production Monitoring

Pipeline:

```text
Feature
├──── Drift
└──── Inference
        ↓
    Monitoring
```

Deliverables:

* monitoring_report

Status:

Completed.

---

# Phase 3.3

## Alerting

Pipeline:

```text
Monitoring
↓
Alerting
```

Deliverables:

* alert_report

Status:

Completed.

---

# Phase 3.4

## Human-in-the-loop Retraining

Workflow:

```text
Alert
↓
Human approval
↓
Training pipeline
↓
Update Champion Registry
```

Purpose:

Allow operators to control retraining.

Status:

Next phase.

---

# Phase 3.5

## Automatic Retraining

Workflow:

```text
Alert
↓
Auto retraining
↓
Evaluate
↓
Register
↓
Update Champion Registry
```

Purpose:

Fully autonomous ML lifecycle.

Status:

Planned.

---

# Phase 3.6

## Champion / Challenger

Workflow:

```text
Champion
│
├── Challenger A
│
├── Challenger B
│
└── Challenger C
```

Purpose:

Safe model replacement.

Status:

Planned.

---

# Phase 4

## Online Serving

Future architecture:

```text
REST API
↓
Model Service
↓
Champion Model
```

Possible technologies:

* FastAPI
* BentoML
* Triton

Status:

Planned.

---

# Phase 4.1

## Real-time Monitoring

Components:

* Prometheus
* Grafana

Status:

Planned.

---

# Phase 4.2

## Feature Store

Possible technologies:

* Feast
* Redis

Status:

Planned.

---

# Phase 4.3

## CI/CD

Possible tools:

* GitHub Actions
* GitLab CI
* Jenkins

Status:

Planned.

---

# Phase 4.4

## Canary Deployment

Workflow:

```text
90% Champion
10% Challenger
```

Purpose:

Safe rollout.

Status:

Planned.

---

# Phase 4.5

## A/B Testing

Workflow:

```text
Traffic
├── Model A
└── Model B
```

Purpose:

Compare business performance.

Status:

Planned.

---

# Phase 5

## Enterprise MLOps

Possible additions:

* Feature Store
* Metadata Store
* Multi-model serving
* Multi-tenant architecture
* Approval workflow
* Governance
* Audit trail
* Explainability
* LLM integration

Status:

Long-term vision.

---

# Philosophy

The project prioritizes:

* reproducibility;
* maintainability;
* loose coupling;
* deterministic behavior;
* production readiness;

over quick proof-of-concept solutions.

The architecture is intentionally designed to evolve gradually without requiring large refactoring.
