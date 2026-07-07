# Resource Cleanup Guide

## Overview

The `delete_tasks.py` utility safely removes all ClearML tasks, models, and projects related to the cashflow forecasting pipeline.

Use this to:

* Clean up after development/testing
* Free server storage
* Prepare for fresh runs
* Debug project hierarchy issues

---

## Prerequisites

* ClearML SDK installed: `pip install clearml`
* Credentials configured: `~/.clearml/clearml.conf`
* API access to target project

---

## Usage

### Safe Cleanup (with confirmation)

```bash
python delete_tasks.py
```

**Prompts:**

1. List of projects to delete
2. Confirmation before deletion
3. Progress updates

**Output:**

```
Projects to delete:
  * Deposit-CashFlow/Templates
  * Deposit-CashFlow/Pipelines
  * Deposit-CashFlow/Datasets
  * Deposit-CashFlow/Registry

WARNING: This will permanently delete all tasks, models, and datasets.
Continue? (yes/no): yes

Deleting models...
  ✓ 15 models deleted
Deleting tasks...
  ✓ 127 tasks deleted
Deleting projects...
  ✓ Deposit-CashFlow/Templates deleted
  ✓ Deposit-CashFlow/Pipelines deleted
  ✓ Deposit-CashFlow/Datasets deleted
  ✓ Deposit-CashFlow/Registry deleted

Cleanup complete!
```

### Dry Run (preview only)

```bash
python delete_tasks.py --dry-run
```

Lists what would be deleted without actually deleting.

---

## Cleanup Strategy

### Option 1: Full Cleanup (Recommended)

Delete all cashflow projects:

```bash
python delete_tasks.py
```

**What gets deleted:**

* All task templates
* All training/production runs
* All models (candidate/champion)
* All feature datasets
* All registry tasks
* Project hierarchy

**When to use:**

* Cleaning up after testing
* Removing old pipeline runs
* Fresh start for new development

### Option 2: Selective Cleanup

Delete specific project only:

```python
# Edit delete_tasks.py, modify target projects:
TARGET_PROJECTS = [
    "Deposit-CashFlow/Pipelines",  # Only delete pipelines
]
```

Then run:

```bash
python delete_tasks.py
```

### Option 3: Manual Cleanup via ClearML Web UI

1. Go to Projects
2. Right-click project
3. Delete project
4. Confirm

**Limitation:** Cannot delete projects with datasets using UI alone (must use `delete_tasks.py`)

---

## Understanding the Cleanup Process

### Step 1: Delete Models

Removes all trained models:

```text
Models to delete:
  ✓ candidate_model_v1
  ✓ candidate_model_v2
  ✓ champion_model_v1
```

**What happens:**

* Model artifacts removed
* Model metadata cleared
* Tags (candidate/champion) removed

### Step 2: Delete Tasks

Removes all task executions:

```text
Tasks to delete:
  ✓ extract (127 runs)
  ✓ feature (127 runs)
  ✓ train (89 runs)
  ... (total 1,234 tasks)
```

**What's NOT deleted:**

* Task templates (definitions)
* Task output logs (archived separately)

### Step 3: Delete Datasets

Removes all feature datasets:

```text
Datasets to delete:
  ✓ cashflow_features_v1 (12 versions)
  ✓ cashflow_features_v2 (8 versions)
```

**Critical:** Uses `force=True` to delete versioned datasets

### Step 4: Delete Projects

Removes project folders:

```text
Projects to delete:
  ✓ Deposit-CashFlow/Templates
  ✓ Deposit-CashFlow/Pipelines
  ✓ Deposit-CashFlow/Datasets
  ✓ Deposit-CashFlow/Registry
```

**Result:** Clean project hierarchy ready for fresh runs

---

## Common Scenarios

### Scenario 1: Templates Appear at Wrong Level

**Symptom:** Templates show as `Deposit-CashFlow > Templates` instead of under parent

**Cause:** Old tasks with incorrect naming still exist

**Solution:**

```bash
python delete_tasks.py  # Full cleanup
python register_templates.py  # Re-register fresh
```

### Scenario 2: Cannot Delete Dataset (Project Not Empty)

**Symptom:** Error `Project has associated non-empty datasets`

**Solution:** `delete_tasks.py` automatically uses `force=True`:

```python
client.projects.delete(project=project_id, force=True)
```

If manual cleanup needed:

```python
from clearml import Dataset
Dataset.delete(dataset_id=dataset_id, force=True)
```

### Scenario 3: Partial Cleanup (Keep Some Models)

**Scenario:** Keep champion model, delete others

**Steps:**

1. Note champion model ID
2. Edit `delete_tasks.py`:

   ```python
   # Skip champion model
   if model["id"] == "champion_model_id":
       continue
   ```

3. Run cleanup

### Scenario 4: Large Project (>10K Tasks)

**Performance:** Cleanup may take 10-30 minutes

**Optimization:**

```bash
# Run in background
nohup python delete_tasks.py > cleanup.log 2>&1 &

# Monitor progress
tail -f cleanup.log
```

---

## Safety Considerations

### Backup Before Cleanup

ClearML stores task data in database. Backup if needed:

```bash
# Export project data (via ClearML API)
from clearml.backend_api.session.client import APIClient

client = APIClient()
tasks = client.tasks.get_all(project="Deposit-CashFlow/Pipelines")
# Export to JSON for archival
```

### Confirm Before Deleting

Always use `--dry-run` first:

```bash
python delete_tasks.py --dry-run
# Review output, then:
python delete_tasks.py
```

### Undo Is Impossible

Deleted tasks and models **cannot be recovered**. Be absolutely sure before confirming.

### What Survives Cleanup

* ClearML server configuration
* User accounts and credentials
* Queues and agent definitions
* Local `.env` file
* Git repository

---

## Post-Cleanup Verification

After running cleanup, verify:

### 1. Check Projects

```bash
# Via ClearML Web UI
# Projects tab - should only show new runs
```

### 2. Count Remaining Items

```bash
clearml-task list --project "Deposit-CashFlow*"  # Should return few/none
```

### 3. Re-register Templates

```bash
python register_templates.py
# Should register 15 new templates
```

### 4. Verify Project Hierarchy

```
Deposit-CashFlow
├── /Templates       (15 new templates)
├── /Pipelines       (empty)
├── /Datasets        (empty)
└── /Registry        (empty)
```

---

## Troubleshooting

### Issue: "Connection refused" during cleanup

**Symptom:** Cleanup fails to connect to server

**Solution:**

```bash
# Verify server is running
curl http://localhost:8080

# Or specify server in command
CLEARML_WEB_HOST=http://clearml.company.com:8080 python delete_tasks.py
```

### Issue: "Permission denied" errors

**Symptom:** Cleanup fails with permission errors

**Solution:**

```bash
# Verify credentials
cat ~/.clearml/clearml.conf | grep api

# Or re-initialize
clearml-agent init
```

### Issue: Cleanup hangs after 5 minutes

**Symptom:** Process stops responding

**Solution:**

```bash
# Increase timeout
CLEARML_TIMEOUT=300 python delete_tasks.py

# Or use with screen/tmux for background execution
screen -S cleanup
python delete_tasks.py
# Ctrl-A then D to detach
```

---

## Advanced: Custom Cleanup Script

For selective deletion, create custom script:

```python
from clearml.backend_api.session.client import APIClient

client = APIClient()

# Delete only old runs (>30 days)
from datetime import datetime, timedelta

cutoff = datetime.now() - timedelta(days=30)
tasks = client.tasks.get_all(
    project="Deposit-CashFlow/Pipelines",
    created={"<": cutoff.isoformat()}
)

for task in tasks:
    client.tasks.delete(task=task["id"])
    print(f"Deleted {task['name']}")
```

---

## Performance Tips

### Parallel Deletion

For large cleanup, parallelize:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    for task_id in task_ids:
        executor.submit(client.tasks.delete, task=task_id)
```

### Batch Operations

Delete in batches to reduce API calls:

```python
# Instead of: delete each model individually
client.models.delete(model=model_id)

# Do: Batch delete
models_to_delete = [m["id"] for m in models]
# ClearML API doesn't support batch, but threading helps
```

---

## Integration with CI/CD

### Cleanup Before Fresh Run

In GitHub Actions:

```yaml
- name: Cleanup old runs
  run: python delete_tasks.py --dry-run  # Remove --dry-run when confident

- name: Register templates
  run: python register_templates.py

- name: Run pipelines
  run: python pipelines/trainning_pipeline.py
```

### Schedule Cleanup

Cron job for automatic cleanup:

```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/cashflow-clearml && python delete_tasks.py
```

---

## References

* [ClearML APIClient Documentation](https://clear.ml/docs/latest/docs/references/sdk/api_client/)
* [Project Management](https://clear.ml/docs/latest/docs/fundamentals/projects/)
* [Dataset Management](https://clear.ml/docs/latest/docs/fundamentals/data_repo_storage/)
