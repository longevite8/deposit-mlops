# ClearML Pipeline Cache Scenario Results

## Run Context

| Field | Value |
| --- | --- |
| Test started | 2026-07-27T16:38:36+07:00 |
| Experiment branch | `test-clearml-cache-behavior` |
| Baseline branch before experiment | `test` |
| `COMMIT_A` baseline | `5fc67228ad533d01b8f12df992fc2f33ded38cc1` |
| ClearML server URL | Pending ClearML run |
| CPU queue | Pending ClearML run |
| Services queue | Pending ClearML run |

## Cache Configuration

### Training Pipeline

All training steps were configured with `cache_executed_step=True` for this experiment.

| Step | Cache enabled |
| --- | --- |
| `extract` | `true` |
| `feature` | `true` |
| `validate` | `true` |
| `drift` | `true` |
| `hpo` | `true` |
| `train` | `true` |
| `evaluate` | `true` |
| `register` | `true` |
| `deploy_candidate_serving` | `true` |
| `verify_candidate_endpoint` | `true` |
| `explain_model` | `true` |
| `compare_champion` | `true` |
| `promote_champion` | `true` |
| `deploy_serving` | `true` |
| `verify_endpoint` | `true` |

### Production Pipeline

Production cache settings were intentionally left unchanged.

| Step | Cache enabled |
| --- | --- |
| `extract` | `false` |
| `feature` | `false` |
| `drift` | `false` |
| `inference` | `false` |
| `monitoring` | `false` |
| `alerting` | `false` |
| `auto_retraining` | `false` |

## Validation Log

| Check | Result | Notes |
| --- | --- | --- |
| Pre-change `uv run python -m unittest tests.test_pipeline_specs` | Passed | 7 tests passed before cache edits. |
| Post-change `uv run python -m unittest tests.test_pipeline_specs` | Passed | 7 tests passed after all training steps were made cacheable. |
| Manifest/cache inspection | Passed | Python inspection confirmed all training steps are cacheable and production remains uncached. |

## Git Commits

| Label | Commit | Description |
| --- | --- | --- |
| `COMMIT_A` | `5fc67228ad533d01b8f12df992fc2f33ded38cc1` | Baseline before experiment branch edits. |
| All-cached config commit | Pending | Commit containing `pipelines/specs.py`, test, and this result log. |
| `COMMIT_B` | Pending | Harmless `tasks/train_model.py` trace change with template re-registration. |
| `COMMIT_C` | Pending | Harmless `tasks/train_model.py` trace change without template re-registration. |
| Failed train commit | Pending | Controlled failure in `tasks/train_model.py`. |
| Fixed train commit | Pending | Recovery commit for failed-train scenario. |

## Template Registrations

Pending real ClearML run.

| Scenario | Template name | Template ID | Script path | Recorded commit |
| --- | --- | --- | --- | --- |
| Scenario 1 | Pending | Pending | Pending | Pending |

## Scenario 1: Baseline Successful Pipeline

Purpose: establish successful cached candidates.

| Field | Value |
| --- | --- |
| Commands | Pending |
| Pipeline controller task ID | Pending |
| Pipeline terminal status | Pending |
| Expected result | First clean run should execute steps because no prior identical successful cache exists for this setup. |
| Actual result | Pending |

| Step | Task ID | Status | Git/script evidence | Executed or reused | Key artifacts |
| --- | --- | --- | --- | --- | --- |
| `extract` | Pending | Pending | Pending | Pending | Pending |
| `feature` | Pending | Pending | Pending | Pending | Pending |
| `validate` | Pending | Pending | Pending | Pending | Pending |
| `drift` | Pending | Pending | Pending | Pending | Pending |
| `hpo` | Pending | Pending | Pending | Pending | Pending |
| `train` | Pending | Pending | Pending | Pending | Pending |
| `evaluate` | Pending | Pending | Pending | Pending | Pending |
| `register` | Pending | Pending | Pending | Pending | Pending |
| `deploy_candidate_serving` | Pending | Pending | Pending | Pending | Pending |
| `verify_candidate_endpoint` | Pending | Pending | Pending | Pending | Pending |
| `explain_model` | Pending | Pending | Pending | Pending | Pending |
| `compare_champion` | Pending | Pending | Pending | Pending | Pending |
| `promote_champion` | Pending | Pending | Pending | Pending | Pending |
| `deploy_serving` | Pending | Pending | Pending | Pending | Pending |
| `verify_endpoint` | Pending | Pending | Pending | Pending | Pending |

## Scenario 2: Change Only Training Logic

Purpose: prove upstream unchanged steps cache while changed training step reruns.

| Field | Value |
| --- | --- |
| Commands | Pending |
| Pipeline controller task ID | Pending |
| Pipeline terminal status | Pending |
| Expected result | Upstream unchanged steps cache; `train` reruns after template update; downstream steps rerun when fed by new `train`. |
| Actual result | Pending |

| Step | Scenario 1 task ID | Scenario 2 task ID | Expected | Actual |
| --- | --- | --- | --- | --- |
| `extract` | Pending | Pending | Reused | Pending |
| `feature` | Pending | Pending | Reused | Pending |
| `validate` | Pending | Pending | Reused | Pending |
| `drift` | Pending | Pending | Reused | Pending |
| `hpo` | Pending | Pending | Reused | Pending |
| `train` | Pending | Pending | Rerun | Pending |
| downstream | Pending | Pending | Rerun as affected by `train` | Pending |

## Scenario 3: Commit Without Re-Registering Templates

Purpose: prove whether commit alone changes ClearML step identity in this repo.

| Field | Value |
| --- | --- |
| Commands | Pending |
| Pipeline controller task ID | Pending |
| Pipeline terminal status | Pending |
| Expected result | Environment-specific: template-pinned tasks may keep old code, while dynamic branch HEAD may rerun. |
| Actual result | Pending |

## Scenario 4: Failed Train, New Pipeline Recovery

Purpose: test the recommended recovery workflow.

| Field | Value |
| --- | --- |
| Failed pipeline controller task ID | Pending |
| Recovery pipeline controller task ID | Pending |
| Expected result | Upstream successful steps are reused; fixed `train` reruns; downstream steps continue. |
| Actual result | Pending |

## Scenario 5: Same Failed Pipeline Manual Repair

Purpose: verify whether editing the failed child task Git info can resume the same pipeline.

| Field | Value |
| --- | --- |
| Failed pipeline reused | Pending |
| Manual task repair action | Pending |
| Expected result | Child task may succeed; parent pipeline may not resume cleanly. |
| Actual result | Pending |

## Final Conclusion

Pending real ClearML run.

## Operational Recommendation

Pending real ClearML run.
