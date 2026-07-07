# Champion Registry

## Overview

The system uses a singleton Champion Registry to track the current production model.

Instead of searching for the latest model, the production pipeline always reads the champion model from this registry.

This guarantees deterministic inference.

---

# Motivation

Without a registry, inference would require:

```python
Model.query_models()
```

or

```python
Task.get_tasks()
```

which introduces uncertainty.

Examples:

* newest task may not be the best model;
* multiple models may exist;
* rollback becomes difficult.

The Champion Registry solves these issues.

---

# Architecture

```text id="wjlwmc"
Training Pipeline

Register
↓
Update Champion Registry

----------------------------

Production Pipeline

Load Champion Registry
↓
Inference
```

---

# Singleton Design

Only one registry task exists.

```text id="nhf0rs"
Project:
Cashflow Registry

Task:
Champion Registry
```

This task lives independently from pipelines.

Its task_id never changes.

---

# Bootstrap

The registry task is created only once.

Example:

```python
from clearml import Task

task = Task.init(
    project_name="Cashflow Registry",
    task_name="Champion Registry",
    task_type=Task.TaskTypes.application,
)

task.upload_artifact(
    "champion_registry",
    {
        "version": 0,
        "champion_model_id": "",
        "champion_train_task_id": "",
        "mape": None,
        "r2": None,
    },
)

print(task.id)

task.close()
```

After creating the task, store:

```python
CHAMPION_REGISTRY_TASK_ID
```

inside:

```text id="t38kkt"
config.py
```

---

# Registry Structure

Example:

```python id="ntz0fy"
champion_registry = {
    "version": 12,
    "champion_model_id": "...",
    "champion_train_task_id": "...",
    "mape": 0.031,
    "r2": 0.94,
}
```

---

# Update Champion Registry

Training pipeline:

```text id="a8hs7z"
Register
↓
Update Champion Registry
```

The update task writes:

```python id="2ln6b4"
champion_registry
```

into the singleton task.

---

# Load Champion Registry

Production pipeline:

```text id="fjmbwf"
Load Champion Registry
↓
Inference
```

Load task reads:

```python
registry_task = Task.get_task(
    task_id=CHAMPION_REGISTRY_TASK_ID
)

registry = registry_task.artifacts[
    "champion_registry"
].get()
```

and exposes:

```python id="twup1i"
champion_registry
```

to downstream tasks.

---

# Inference

Inference consumes:

* feature_df
* champion_registry

and loads:

```python
champion_model_id
```

from:

```python
champion_registry
```

Example:

```python id="sr7k1h"
champion_model = Model(
    model_id=champion_model_id
)

model_path = champion_model.get_local_copy()

model = joblib.load(model_path)
```

---

# Versioning

Each promotion increments:

```python
version
```

Example:

```python id="djhc6n"
version = 13
```

This enables:

* history tracking;
* rollback;
* auditability.

---

# Rollback

Rollback simply means updating:

```python id="g65rle"
champion_model_id
```

to a previous model.

No retraining is required.

Example:

```python id="7p8gxp"
champion_registry = {
    "version": 14,
    "champion_model_id": previous_model_id,
    ...
}
```

---

# Deterministic Production

Inference never uses:

```python id="gb44dr"
Model.query_models()
```

or:

```python id="1lmh33"
Task.get_tasks()
```

Therefore:

* production runs are reproducible;
* model selection is explicit;
* behavior is predictable.

---

# Future Champion / Challenger

Current architecture:

```text id="dd53z8"
Champion
```

Future architecture:

```text id="ptjlwm"
Champion
│
└── Challenger
```

Registry may evolve into:

```python id="h24ej6"
champion_registry = {
    "version": 20,

    "champion_model_id": "...",

    "challenger_model_id": "...",

    "champion_train_task_id": "...",

    "challenger_train_task_id": "...",

    "mape": ...,

    "r2": ...,
}
```

without changing the production pipeline.

---

# Human-in-the-loop Approval

Future workflow:

```text id="l95y74"
Register
↓
Manual Approval
↓
Update Champion Registry
```

Models are promoted only after approval.

---

# Automatic Retraining

Future workflow:

```text id="7c33zs"
Drift detected
↓
Retraining Pipeline
↓
Evaluate
↓
Register
↓
Update Champion Registry
```

---

# Benefits

The Champion Registry provides:

* deterministic inference;
* model versioning;
* rollback capability;
* auditability;
* reproducibility;
* Champion / Challenger support;
* future A/B testing support;
* minimal coupling.

---

# Design Philosophy

The Champion Registry acts as a lightweight internal model registry.

It plays a role similar to:

* MLflow Model Registry;
* SageMaker Model Registry;
* Vertex AI Model Registry;

while remaining simple and fully integrated with ClearML.
