# ClearML Pipeline Cache Scenario Results

## Run Context

| Field | Value |
| --- | --- |
| Test started | 2026-07-27T16:38:36+07:00 |
| Experiment branch | `test-clearml-cache-behavior` |
| Baseline branch before experiment | `test` |
| ClearML server URL | `http://192.168.140.248:8080` |
| CPU queue | `training` |
| Services queue | `mco-services` |
| Final local status | Worktree clean; branch pushed to `origin/test-clearml-cache-behavior`. |

## Cache Configuration

All `TRAINING_STEPS` were configured with `cache_executed_step=True` for this experiment. `PRODUCTION_STEPS` were left unchanged and remained uncached.

| Pipeline | Step count | Cache result |
| --- | ---: | --- |
| Training | 15 base specs, expanded to 29 child tasks for horizons `7,14,30` | All cache-enabled |
| Production | 7 specs | All cache-disabled |

## Local Validation

| Check | Result | Notes |
| --- | --- | --- |
| Pre-change `uv run python -m unittest tests.test_pipeline_specs` | Passed | 7 tests passed before cache edits. |
| Post-change `uv run python -m unittest tests.test_pipeline_specs` | Passed | 7 tests passed after all training specs were made cacheable. |
| Cache inspection command | Passed | Confirmed every training step had `cache_executed_step=True`; production remained `False`. |

## Git Commits

| Label | Commit | Description |
| --- | --- | --- |
| `COMMIT_A` | `5fc67228ad533d01b8f12df992fc2f33ded38cc1` | Baseline before experiment branch edits. |
| All-cached config commit | `46d4df5042bfb869a554a4198daa4bb2749a93da` | Made every training step cacheable; added this result file. |
| Scenario 1 template fallback commit | `bf839f8631d6f22387d1605eff2c8a2fa14884e1` | Stored Scenario 1 template IDs in `config.py` for remote agents. |
| `COMMIT_B` train marker | `11da2c291dfe908e27461de64ca634392206c5cf` | Added harmless train trace marker. |
| Scenario 2 template fallback commit | `7fa8aad445a277a347f85fee2b47b9800ecb0470` | Stored Scenario 2 template IDs in `config.py`; Scenario 2 train tasks executed from this branch HEAD. |
| `COMMIT_C` no-template-refresh marker | `0ea3b1287d6d08b1e4c987675545d95330e29658` | Changed only train marker and did not re-register templates. |
| Failed train commit | Not executed | Queue/time blocker before controlled-failure scenario. |
| Fixed train commit | Not executed | Queue/time blocker before controlled-failure scenario. |

## Template Registrations

### Scenario 1 Templates

Registered with:

```bash
env GIT_BRANCH=test-clearml-cache-behavior uv run python register_templates.py
```

| Template | ID |
| --- | --- |
| Extract Data | `7034b2119ab946a1a8aaf01918dfedaf` |
| Feature Engineering | `2993113721bc4ffda8728940f29db099` |
| Validate Data | `3233bedf2f7f4e8b82f423ea7cbcdeb7` |
| Drift Detection | `2a0200dcf6f149b68f74710e0e4df2b9` |
| HPO Model | `2bea0fdf45574adc85defd2c7664db39` |
| Train Model | `78ecc3cdb3e243c9a50d2540462c8760` |
| Evaluate Model | `b5eb901773844e90a81c95159c3addac` |
| Register Model | `4f06f43fe9324b18bc77c0e7a6bfa07d` |
| Deploy Candidate Serving | `69b95d25dabe42dfa25d26873301a1ed` |
| Verify Candidate Endpoint | `63af3a1c6fe945c483d7839ee947d2fb` |
| Explain Model | `ad2122287ada4243bb59329119877afb` |
| Compare Champion | `e577681921ed46748818cde8b6ed0d39` |
| Promote Champion | `d6dbbd6db51543519b1f7419e37f41e0` |
| Deploy Serving | `ea4f007faec04bce9978096e022efcba` |
| Verify Endpoint | `11d3339e65384b3bae63fa1cd263d4ef` |

### Scenario 2 Templates

Registered after the train marker commit with the same command.

| Template | ID |
| --- | --- |
| Extract Data | `ef8f0b2c46a24992b2148be808b96152` |
| Feature Engineering | `72e62cd11ac14ce2b5c1afd5f2d13eff` |
| Validate Data | `0e04d22044024c02b3907e2693a2042b` |
| Drift Detection | `ef0f33ce2aaa4052979d1ed4cd003a5d` |
| HPO Model | `b72f4cc744a04fa2949c5ee6f64290e6` |
| Train Model | `e5b2461df5ee4255b7d24ad056116dab` |
| Evaluate Model | `b2c296d5984c4e15be9578187596b16d` |
| Register Model | `993f94f8c80349f393722f999f75ba38` |
| Deploy Candidate Serving | `6bf3a503b5f44b89a0ca1aa25d082a83` |
| Verify Candidate Endpoint | `2a8436887c714f25b05ed7cd5223b56b` |
| Explain Model | `7fc2a1271760456e80cbb2eaf4416470` |
| Compare Champion | `8ca0923d1ca143dc854b10d137579a0b` |
| Promote Champion | `e716b186d4c24668b4dab727d3e86d6d` |
| Deploy Serving | `6897fdbcaecc41d397608e27a3b54339` |
| Verify Endpoint | `c0fb748984234a3c98a48415ed47303d` |

## Scenario 1: Baseline Pipeline

### Attempt 1

| Field | Value |
| --- | --- |
| Command | `uv run python -m pipelines.training_pipeline` |
| Controller ID | `7418dfa05d6f423aa4531d4e3a485d56` |
| Result | Discarded partial attempt |
| Reason | Local-controller mode blocks in `pipe.start_locally(...).wait()`. It was interrupted; the controller stayed `in_progress` with only the first 5 children created. |

### Attempt 2

| Field | Value |
| --- | --- |
| Command | `env RUN_PIPELINE_CONTROLLER_LOCALLY=false uv run python -m pipelines.training_pipeline` |
| Controller ID | `6e0c510e15f34a0d91748e3c487e4686` |
| Result | Failed before child creation |
| Reason | Remote agent had no local `.env`; `config.py` fallback IDs were missing `TEMPLATE_DEPLOY_CANDIDATE_SERVING_ID` and `TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_ID`. |

### Attempt 3

| Field | Value |
| --- | --- |
| Command | `env RUN_PIPELINE_CONTROLLER_LOCALLY=false uv run python -m pipelines.training_pipeline` |
| Controller ID | `a421b707514b44538f1a1c5dfbe523ab` |
| Result | Core baseline completed through train/evaluate/register; stopped due services-queue deadlock |
| Reason stopped | Remote controller occupied `mco-services` while deploy steps were queued to `mco-services`, leaving no free services worker. |

| Step | Task ID | Status at capture |
| --- | --- | --- |
| `extract` | `51b0b6ec00fe4989be2e424039d7475c` | completed |
| `feature` | `6d30946ee1c64774a0503b322a178e6b` | completed |
| `validate` | `b1a8bf7143fc43949ef68286bbbcf1cb` | completed |
| `drift` | `29e19dd22cb744caa7d3a3af8168f1a2` | completed |
| `hpo` | `224a2e0ce8b64857ad8a4c1d671ee2ce` | completed |
| `train_h7` | `20b3d08491a4456fb4ac67546be50791` | completed |
| `train_h14` | `de7d093b4f45407188043063a6429949` | completed |
| `train_h30` | `ac6882fd1421499cadefff457393f030` | completed |
| `evaluate_h7` | `72f6e3b22b424fdbba41f74ede604614` | completed |
| `evaluate_h14` | `2f986ea16e7444238cd7fa0206c353f9` | completed |
| `evaluate_h30` | `a8ebecd0da5846fea5caa06b77c23af9` | completed |
| `register_h7` | `e482cc4cc26544f799d4b38faaac027f` | completed |
| `register_h14` | `6a394bba26bf45bbb6fee393a8f61d7d` | completed |
| `register_h30` | `9c5e919090cb4e2dabadc5c6955a34ce` | completed |
| deploy/explain/compare/promote tail | Multiple | queued/in_progress/completed mixed; run stopped before terminal completion. |

## Scenario 2: Change Only Training Logic and Re-Register Templates

| Field | Value |
| --- | --- |
| Train marker | `CACHE_EXPERIMENT_TRACE = "scenario_2_train_logic_marker"` |
| Re-registered templates | Yes |
| Controller ID | `5eca3ab82df24de9a6eefe0d3e096753` |
| Controller mode | Local controller |
| Result | Core train evidence captured; run stopped before evaluation/deploy tail completed. |

| Step | Scenario 1 task ID | Scenario 2 task ID | Actual result |
| --- | --- | --- | --- |
| `extract` | `51b0b6ec00fe4989be2e424039d7475c` | `a81e2642eeaf4f2f83eb240153e7731d` | Executed as a new task; log showed `Running task id`. |
| `feature` | `6d30946ee1c64774a0503b322a178e6b` | `003c8fc650b74e6093a67a2bbb41d9dd` | Executed as a new task; log showed `Running task id`. |
| `validate` | `b1a8bf7143fc43949ef68286bbbcf1cb` | `91c0d6bb7ea34ab085e7838c79bf06fe` | Executed as a new task; log showed `Running task id`. |
| `drift` | `29e19dd22cb744caa7d3a3af8168f1a2` | `2762c7fcfc534d25aefa5c93a4f4719e` | Executed as a new task. |
| `hpo` | `224a2e0ce8b64857ad8a4c1d671ee2ce` | `fedb1dd62a4c4589ab64222fffdab000` | Executed as a new task. |
| `train_h7` | `20b3d08491a4456fb4ac67546be50791` | `4320d0b67f5547a480fe5ca6f93ffbc3` | Executed and logged marker. |
| `train_h14` | `de7d093b4f45407188043063a6429949` | `f21d70430a6e4c0db71e9ca1dcc0325d` | Executed. |
| `train_h30` | `ac6882fd1421499cadefff457393f030` | `8d1c09d620c74b18b3a385deceab3ef1` | Executed. |

Train evidence from `train_h7`:

```text
commit: 7fa8aad445a277a347f85fee2b47b9800ecb0470
Cache experiment trace: scenario_2_train_logic_marker
```

Conclusion for Scenario 2: because all templates were re-registered after the train change, ClearML created new tasks for unchanged upstream steps too. In this environment, full template re-registration at a new branch HEAD did **not** demonstrate upstream cache reuse.

## Scenario 3: Commit Without Re-Registering Templates

| Field | Value |
| --- | --- |
| Train marker | `CACHE_EXPERIMENT_TRACE = "scenario_3_no_template_reregister_marker"` |
| Re-registered templates | No |
| Controller ID | `c206b680844b4b8a965f0eecb86b6504` |
| Result | Blocked before train evidence |
| Observed state | Controller `in_progress`; child `extract` task `7ccac25e86904d3980fa10539fd6e0be` remained `queued`. |
| Action taken | Local controller interrupted and marked stopped. |

No conclusion was drawn for Scenario 3 because the pipeline did not reach `train`.

## Scenario 4: Failed Train, New Pipeline Recovery

Not executed in this run.

Reason: Scenario 1/2/3 exposed queue/orchestration constraints that made additional controlled-failure runs unsafe within this session. Running this scenario cleanly requires either a free `training` worker and at least one free `mco-services` worker, or a reduced pipeline that stops after `train`.

## Scenario 5: Same Failed Pipeline Manual Repair

Not executed in this run.

Reason: Scenario 4 did not create a controlled failed train task to repair. This should be tested only after Scenario 4 can reliably reach the intended failure point.

## Final Conclusion

- All training steps were successfully configured with `cache_executed_step=True`.
- Real ClearML template registration and pipeline creation worked.
- Remote controller mode exposed an important operational problem: if the controller runs on `mco-services` and deploy steps also run on `mco-services`, a single services worker can deadlock the pipeline tail.
- Scenario 2 showed that **re-registering every template after a train-only code change caused upstream steps to execute as new tasks**, not visibly reuse Scenario 1 tasks.
- The train-only marker did execute, proving the refreshed template/branch path reached the training code.
- The planned “commit without re-registering templates” result is still inconclusive because Scenario 3 stayed queued before `extract` could run.

## Operational Recommendation

- For a clean cache experiment, run a reduced training pipeline that stops after `train`/`evaluate`, or ensure enough workers exist for both the controller and service steps.
- Do not run the controller remotely on the same single-worker services queue used by downstream service tasks.
- To test whether unchanged upstream code caches across Git commits, avoid re-registering all templates. Re-register only the changed template, or create a template-registration helper that can update one template at a time.
- For normal recovery after a failed train task, prefer a new pipeline run with updated template identity rather than manually patching an already-created child task.
