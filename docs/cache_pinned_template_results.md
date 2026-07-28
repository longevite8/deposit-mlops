# ClearML Commit-Pinned Template Cache Results

Test date: 2026-07-28, Asia/Ho_Chi_Minh.

Branch: `test-clearml-cache-behavior`

ClearML server: `http://192.168.140.248:8080`

Queues used:

- Controller/services queue: `mco-services`
- Child CPU queue: `training`

This document records the implementation and real ClearML evidence for reducing wasted compute when `cache_executed_step=True` is used across the training pipeline. The main fix is to pin ClearML task templates to exact Git commits and re-register only templates whose task code changed.

## Implementation

Implemented commits:

| Commit | Purpose |
| --- | --- |
| `43f6e75818fb293eae3c02449b95b521478dae70` | Add commit-pinned template registration by default, add `--branch-head` opt-out, add `--only` targeted registration support, add `TRAINING_PIPELINE_STOP_AFTER`, keep pure training steps cached, keep deploy/verify side-effect steps uncached. |
| `bb5d6c6f5315e07e66d54a4e26cfb5dd0491eebf` | Scenario B trace-only train code change. |
| `4810b207c1a9f93a93a59a2f05e390d8da40554a` | Scenario C train code change without template refresh. |
| `4afdbac10ab401979889369c995da8215ece34a2` | Scenario D controlled failing train commit. |
| `454dae7a4fdbdbf4f2cefece56ca3da1571c2420` | Scenario D fixed train recovery commit. |
| `2aff9cf5e049d0fffd185a618f0c6bc1abe21603` | Clean train task after scenarios. |

Code changes:

- `register_templates.py` now passes `commit=<current git sha>` into `Task.create(...)` by default.
- `register_templates.py --branch-head` or `CLEARML_TEMPLATE_BRANCH_HEAD=true` restores the old branch-head behavior.
- `register_templates.py --only <selector>` registers only selected templates. This is the operational control that avoids invalidating unrelated cached steps.
- `TRAINING_PIPELINE_STOP_AFTER=<step>` allows bounded scenario runs such as `register` without deploying serving endpoints.
- Training deploy/verify side-effect steps remain `cache_executed_step=False`.

Focused validation:

```bash
uv run python -m unittest tests.test_register_templates tests.test_pipeline_specs
```

Result after implementation and after temporary scenario edits: `Ran 19 tests ... OK`.

## Cache Configuration

Training pipeline cache policy after implementation:

| Step | Cache setting | Reason |
| --- | --- | --- |
| `extract` | `True` | Pure upstream data extraction for a fixed input/config should be reusable. |
| `feature` | `True` | Pure feature generation should be reusable when extract output and parameters are identical. |
| `validate` | `True` | Validation result is deterministic for the same feature dataset/config. |
| `drift` | `True` | Drift report is reusable for the same feature dataset/reference config. |
| `hpo` | `True` | HPO result is reusable when inputs and parameters are identical. |
| `train` | `True` | Training result is reusable when task code, inputs, parameters, and environment identity match. |
| `evaluate` | `True` | Evaluation is reusable for the same model/features/eval config. |
| `register` | `True` | Registration metadata is reusable in bounded test runs. |
| `explain_model` | `True` | Explanation output is reusable for the same registered model/input. |
| `compare_champion` | `True` | Comparison result is reusable for the same candidate/champion inputs. |
| `promote_champion` | `True` | Cached in training spec, but should be reviewed before production use because it can mutate model state. |
| `deploy_candidate_serving` | `False` | Side-effect deployment step. |
| `verify_candidate_endpoint` | `False` | Runtime endpoint check. |
| `deploy_serving` | `False` | Side-effect deployment step. |
| `verify_endpoint` | `False` | Runtime endpoint check. |

All scenario pipeline runs used:

```bash
env FORECAST_HORIZONS=7 \
  TRAINING_PIPELINE_STOP_AFTER=register \
  RUN_PIPELINE_CONTROLLER_LOCALLY=true \
  CLEARML_CPU_QUEUE=training \
  CLEARML_SERVICES_QUEUE=mco-services \
  uv run python -m pipelines.training_pipeline
```

## Template Registrations

Baseline all-template registration at `43f6e75818fb293eae3c02449b95b521478dae70`:

```bash
env GIT_BRANCH=test-clearml-cache-behavior uv run python register_templates.py
```

Output confirmed:

```text
Template execution mode: commit-pinned (43f6e75818fb293eae3c02449b95b521478dae70)
```

Baseline template IDs:

| Template | ID |
| --- | --- |
| Extract | `1cf032df72fe4753baae52c9b30ec75c` |
| Feature | `f8753dae9f2e4b1685f9dbe1b0761bcb` |
| Validate | `f1a4d632696546a582354c3e17f73c1d` |
| Drift | `956970c387614961a638e8ef6af1d909` |
| HPO | `af654f8623154c2fab426755e0646725` |
| Train | `74b5bcd1c2cc4791aa263268a5d43512` |
| Evaluate | `b16e42030b904e5e9d659572a5309c36` |
| Register | `477080fef44b4483bbd7758dec17963e` |
| Compare Champion | `68c99f45cdd748c3bf3667dc51bf84fb` |
| Promote Champion | `b9f76f95819b4f3096203829e3c89965` |
| Explain | `0fde318beda5467abb5366b2b1d11c1d` |
| Deploy Serving | `3d13c4d597ad43a2948b8e1701a21f07` |
| Verify Endpoint | `c81b46d030e7409a95d079fd58e2ef47` |
| Deploy Candidate | `ac82089e9b2a41399b30031f319a4cea` |
| Verify Candidate | `0b6c3a67e22b4721b308b1fa69153c83` |

Train-only template registrations:

| Scenario | Command | Template ID | Pinned commit |
| --- | --- | --- | --- |
| B train change | `env GIT_BRANCH=test-clearml-cache-behavior uv run python register_templates.py --only train` | `4cddc7ec7e1247f3a9712502de34833e` | `bb5d6c6f5315e07e66d54a4e26cfb5dd0491eebf` |
| D failing train | Same command | `59e5d4b03ef841fa9cfc56d587dce755` | `4afdbac10ab401979889369c995da8215ece34a2` |
| D fixed train | Same command | `4729b4dec94441f891aa60c267ee720d` | `454dae7a4fdbdbf4f2cefece56ca3da1571c2420` |
| Final clean train | Same command | `a842da436bb64352a4e549d3e9f786d2` | `2aff9cf5e049d0fffd185a618f0c6bc1abe21603` |

## Scenario A: Baseline Successful Pipeline

Purpose: establish successful cache candidates using commit-pinned templates.

Controller task: `674e38ad65e644dfbf3ca308b5fc5fe4`

Controller status: `completed`

Controller commit: `43f6e75818fb293eae3c02449b95b521478dae70`

Child steps:

| Step | Task ID | Status | Commit | Result |
| --- | --- | --- | --- | --- |
| `extract` | `e563e4f01eb44e37968de4950e46eb0b` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed. |
| `feature` | `5e0f9da5fc0c43f2820fedbe25f24f8b` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed. |
| `validate` | `2039e5d839e044c5b3cc3c9c98045399` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed. |
| `drift` | `538180de33804d1a9dc19800d0ed988f` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed. |
| `hpo` | `a2aeab18834f4be580f92396a2fe2495` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed. |
| `train` | `149867197b7842c293166f3fdafc1eb9` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed. |
| `evaluate` | `14406abb63e74af6adfe2d3579e093b3` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed. |
| `register` | `a2182a43bdc549ac8a3f01fe0375a8b0` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed. |

Observed artifacts:

- `extract`: `raw_data`, `extract_lineage`, `extract_summary`
- `feature`: `feature_dataset_id`, `feature_lineage`, `feature_summary`
- `validate`: `validation_report`, `validate_lineage`, `validate_summary`
- `drift`: `drift_lineage`, `drift_summary`
- `hpo`: `best_params`, `hpo_lineage`, `hpo_summary`
- `train`: `model_archive`, `model_id`, `training_metrics`, `train_summary`, `train_lineage`
- `evaluate`: `forecasts`, `forecast_with_actuals`, `evaluate_summary`, `evaluate_lineage`
- `register`: `register_summary`, `register_lineage`

Actual result: first run executed the bounded core DAG and produced reusable successful task candidates.

## Scenario B: Change Only Training Logic And Re-Register Train

Purpose: prove unchanged upstream steps are cached when only the train template changes.

Code change: trace-only train change committed at `bb5d6c6f5315e07e66d54a4e26cfb5dd0491eebf`.

Template registration:

```bash
env GIT_BRANCH=test-clearml-cache-behavior uv run python register_templates.py --only train
```

Registered train template: `4cddc7ec7e1247f3a9712502de34833e`

Controller task: `067628982d1742ca97cb741c69c297eb`

Controller status: `completed`

Controller commit: `bb5d6c6f5315e07e66d54a4e26cfb5dd0491eebf`

Child steps:

| Step | Task ID | Status | Commit | Result |
| --- | --- | --- | --- | --- |
| `extract` | none under controller | cached | `43f6e75818fb293eae3c02449b95b521478dae70` template remained unchanged | Cache hit inferred because no child task was materialized. |
| `feature` | none under controller | cached | `43f6e75818fb293eae3c02449b95b521478dae70` template remained unchanged | Cache hit inferred because no child task was materialized. |
| `validate` | none under controller | cached | `43f6e75818fb293eae3c02449b95b521478dae70` template remained unchanged | Cache hit inferred because no child task was materialized. |
| `drift` | none under controller | cached | `43f6e75818fb293eae3c02449b95b521478dae70` template remained unchanged | Cache hit inferred because no child task was materialized. |
| `hpo` | none under controller | cached | `43f6e75818fb293eae3c02449b95b521478dae70` template remained unchanged | Cache hit inferred because no child task was materialized. |
| `train` | `362505984c1f40249e5800c4d9a49b06` | `completed` | `bb5d6c6f5315e07e66d54a4e26cfb5dd0491eebf` | Executed because train template identity changed. |
| `evaluate` | `f2507a28c27045bcb581ca4d4e8122eb` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed because `${train.id}` changed. |
| `register` | `457e2a2fe4614b308dc695fb0f17a377` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed because upstream evaluate/train inputs changed. |

Evidence from train child:

```text
Pinned cache experiment trace: scenario_b_train_only_change
Process completed successfully
```

Actual result: unchanged upstream steps were reused from cache. Only `train` and affected downstream steps executed.

## Scenario C: Commit Train Code Without Re-Registering Templates

Purpose: prove whether a Git commit alone changes step identity.

Code change: train marker committed at `4810b207c1a9f93a93a59a2f05e390d8da40554a`.

Template registration: intentionally skipped.

Controller task: `539c02c814e84fae833eb0365d6d851c`

Controller status: `completed`

Controller commit: `4810b207c1a9f93a93a59a2f05e390d8da40554a`

Child steps:

| Step scope | Result |
| --- | --- |
| All child steps through `register` | No child tasks were materialized under the controller. |

Actual result:

- The controller itself ran at `4810b207c1a9f93a93a59a2f05e390d8da40554a`.
- The step templates did not change because `register_templates.py` was not run.
- ClearML completed the pipeline in about 17 seconds with zero child tasks under the controller.
- The Scenario C train marker did not execute.

Conclusion: commit alone is not enough. With commit-pinned templates, the pipeline uses the template IDs currently stored in `.env`. To make a changed task rerun, re-register that task template and update `.env`.

## Scenario D1: Failed Train With Re-Registered Failing Template

Purpose: prove failed train recovery starts from cached upstream tasks.

Code change: controlled train failure committed at `4afdbac10ab401979889369c995da8215ece34a2`.

Template registration:

```bash
env GIT_BRANCH=test-clearml-cache-behavior uv run python register_templates.py --only train
```

Registered failing train template: `59e5d4b03ef841fa9cfc56d587dce755`

Controller task: `40b07de7a17b46318736679216f5aca8`

Controller status: `failed`

Controller commit: `4afdbac10ab401979889369c995da8215ece34a2`

Child steps:

| Step | Task ID | Status | Commit | Result |
| --- | --- | --- | --- | --- |
| Upstream steps through `hpo` | none under controller | cached | baseline templates unchanged | Cache hit inferred because no upstream child task was materialized. |
| `train` | `5f7c15a49a5d42bab31435dab577b90d` | `failed` | `4afdbac10ab401979889369c995da8215ece34a2` | Executed and failed intentionally. |
| `evaluate` | none | not run | n/a | Blocked by failed train. |
| `register` | none | not run | n/a | Blocked by failed train. |

Evidence from train child:

```text
Pinned cache experiment trace: scenario_d_controlled_failure
RuntimeError: Pinned cache experiment controlled train failure.
Process failed, exit code 1
```

Actual result: the failing run did not recompute upstream steps. Only the changed train task executed and failed.

## Scenario D2: Fixed Train, New Pipeline Recovery

Purpose: prove the recommended recovery workflow.

Code change: fixed train recovery committed at `454dae7a4fdbdbf4f2cefece56ca3da1571c2420`.

Template registration:

```bash
env GIT_BRANCH=test-clearml-cache-behavior uv run python register_templates.py --only train
```

Registered fixed train template: `4729b4dec94441f891aa60c267ee720d`

Controller task: `86e248f6fa14418e9407e961e23f9a05`

Controller status: `completed`

Controller commit: `454dae7a4fdbdbf4f2cefece56ca3da1571c2420`

Child steps:

| Step | Task ID | Status | Commit | Result |
| --- | --- | --- | --- | --- |
| Upstream steps through `hpo` | none under controller | cached | baseline templates unchanged | Cache hit inferred because no upstream child task was materialized. |
| `train` | `c07023d9675f411e97712e9836ef83c1` | `completed` | `454dae7a4fdbdbf4f2cefece56ca3da1571c2420` | Executed fixed train. |
| `evaluate` | `c74cb8a5fec54682ba5d7f769024278a` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed because train output changed. |
| `register` | `26fc3725bfd94cb080a8a421fcbf4885` | `completed` | `43f6e75818fb293eae3c02449b95b521478dae70` | Executed after fixed evaluation. |

Evidence from train child:

```text
Pinned cache experiment trace: scenario_d_fixed_recovery
Process completed successfully
```

Actual result: the reliable recovery workflow is to commit the fix, re-register the fixed task template, and start a new pipeline. Upstream successful work was reused.

## Final Cleanup

Temporary markers and failure hooks were removed from `tasks/train_model.py`.

Clean commit: `2aff9cf5e049d0fffd185a618f0c6bc1abe21603`

Final clean train template:

| Template | ID | Pinned commit |
| --- | --- | --- |
| Train | `a842da436bb64352a4e549d3e9f786d2` | `2aff9cf5e049d0fffd185a618f0c6bc1abe21603` |

No final full pipeline was run after cleanup to avoid additional compute. The clean template was registered so future runs do not use scenario-only train code.

## Why The Cache Worked

ClearML pipeline cache for `cache_executed_step=True` uses the step's effective task identity and inputs, not just the Python function name. In this repository, the key identity includes the base task template referenced by `base_task_id`; that template includes execution details such as repository, branch, commit, script entry point, requirements, parameters, and artifacts/inputs supplied through pipeline parameter overrides.

Before this implementation, template registration tracked branch head. When the repository branch moved from commit A to commit B, unchanged tasks could still look different to ClearML because their execution section resolved to a different Git commit. That invalidated cache even when task code, parameters, and data were effectively unchanged.

After this implementation:

- Unchanged templates remain pinned to their old successful commit.
- `register_templates.py --only train` changes only the train template ID and train pinned commit.
- `extract`, `feature`, `validate`, `drift`, and `hpo` still point to the same baseline template IDs, so ClearML can reuse their successful previous executions.
- `evaluate` and `register` rerun when `train` produces a new task ID, because their upstream input references changed.

This is why Scenario B did not rerun upstream steps but did rerun train and downstream steps.

## Operational Recommendation

Use commit-pinned templates by default.

When task code changes, re-register only the templates for changed task files:

```bash
env GIT_BRANCH=<branch> uv run python register_templates.py --only train
```

Then start a new pipeline:

```bash
env FORECAST_HORIZONS=7 uv run python -m pipelines.training_pipeline
```

Do not re-register every template for every repository commit. A docs-only commit, pipeline-only commit, or unrelated task commit should not force all templates to new commits. If all templates are refreshed on every commit, ClearML can treat unchanged steps as different tasks and waste resources.

For failed pipeline recovery:

1. Fix the failed task code.
2. Commit and push.
3. Re-register only the fixed task template.
4. Start a new pipeline.

Do not rely on editing and re-enqueuing a failed child task inside the same failed pipeline as the normal operational workflow. A new pipeline gives ClearML a clean controller state and still reuses successful cached upstream work.

Keep side-effect runtime steps uncached:

- deployment steps
- endpoint verification steps
- production inference/monitoring/alerting steps

Those steps interact with live services or current runtime state, so saving compute is less important than avoiding stale side effects.

## Decision

The implemented approach solves the resource waste case:

- If `extract` code, environment, parameters, and inputs are unchanged across commits, it stays cached because its template remains pinned to the old successful commit.
- If only `train` changes, re-register only `train`; upstream cache is preserved.
- If a commit is made but templates are not re-registered, the changed task code does not run.
- If a pipeline fails at `train`, a fresh pipeline after fixing and re-registering `train` reuses upstream cache and reruns only the affected path.

