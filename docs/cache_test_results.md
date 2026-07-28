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

---

## Continuation Run: 2026-07-28

This continuation used the same branch, `test-clearml-cache-behavior`, with full control of the local ClearML server/agents. The queue problem from the first run was avoided by running the pipeline controller locally and leaving the `training` and `mco-services` agents free for child tasks.

| Field | Value |
| --- | --- |
| Date/time | 2026-07-28 Asia/Ho_Chi_Minh |
| ClearML server | `http://192.168.140.248:8080` |
| Branch | `test-clearml-cache-behavior` |
| Horizon setting | `FORECAST_HORIZONS=7` |
| Controller mode | local controller, `run_pipeline_steps_locally=False` |
| CPU queue | `training` |
| Services queue | `mco-services` |
| Snapshot helper | `scripts/py/experiments/clearml_cache_snapshot.py` |

### Continuation Commits

| Label | Commit | Notes |
| --- | --- | --- |
| Continuation baseline | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` | Added snapshot helper; current train marker before retry scenarios. |
| Scenario 2 retry | `7c4df03d9c18b889454c4d1849f822d203f08871` | Changed only `tasks/train_model.py` marker and re-registered only Train Model. |
| Scenario 3 retry | `022dca3818c3973ab69d7e68ef4bf335328ad2a0` | Changed only `tasks/train_model.py` marker and did not re-register templates. |
| Scenario 4 failed commit | `922d415ef0beeaa7f2514fcd0faa1c552c06c6f9` | Added controlled train failure. |
| Scenario 4 fixed commit | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` | Disabled controlled failure. |

### Continuation Template Registrations

| Registration | Command | Template ID |
| --- | --- | --- |
| Scenario 2 retry Train Model | `env GIT_BRANCH=test-clearml-cache-behavior uv run python register_templates.py --only train` | `80a1b704eaf2488b92e5dc25152901d8` |
| Scenario 4 failing Train Model | same command | `bd665008567343b59be66c24cddc98b8` |
| Scenario 4 fixed Train Model | same command | `49832be9525e4bb98aa642089d2c71d4` |

Upstream template IDs were intentionally left unchanged during Scenario 2 retry and Scenario 3 retry:

| Template | ID |
| --- | --- |
| Extract Data | `ef8f0b2c46a24992b2148be808b96152` |
| Feature Engineering | `72e62cd11ac14ce2b5c1afd5f2d13eff` |
| Validate Data | `0e04d22044024c02b3907e2693a2042b` |
| Drift Detection | `ef0f33ce2aaa4052979d1ed4cd003a5d` |
| HPO Model | `b72f4cc744a04fa2949c5ee6f64290e6` |

## Continuation Baseline

| Field | Value |
| --- | --- |
| Command | `env FORECAST_HORIZONS=7 RUN_PIPELINE_CONTROLLER_LOCALLY=true CLEARML_CPU_QUEUE=training CLEARML_SERVICES_QUEUE=mco-services uv run python -m pipelines.training_pipeline` |
| Controller ID | `17def6d0e7bd4953a971cc025e978b1c` |
| Controller status | `completed` |
| Controller commit | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| Result | Full 15-step one-horizon training pipeline completed. |

| Step | Task ID | Status | Commit |
| --- | --- | --- | --- |
| extract | `d8871a59ef9f427aadae593ff82ade8d` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| feature | `a1162a52d6104786a8e685d5add7e1d3` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| validate | `cf348eeb117d46bcbcbb6974cd524957` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| drift | `f68d01361db74ca489ca55c6e316dfd3` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| hpo | `cba14ee087744da787a117d7f9e2826b` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| train | `322fdb5748764be6bf80ee55340c1be5` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| evaluate | `69abe032e2f74a2b98ec1ba1520faa44` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| register | `8f242e4266af444cabd8565ef8cfd4b9` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| deploy_candidate_serving | `fbb8095be9634d79a8ccb5ecef315959` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| compare_champion | `d7590b65788c4a47a4cbcc8565802d2a` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| explain_model | `5607dc0f91534748a6570cdd08ca8893` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| promote_champion | `de2b4bbc211946f8ae9a19bf1d5b0099` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| verify_candidate_endpoint | `80a69f662dda4af6b0201a4b277df76f` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| deploy_serving | `a2c60c59875a42d8a45067ad2462d915` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |
| verify_endpoint | `7d128a8c51e74d0eb7b10f51919da1df` | completed | `0ae9c40817adcccf3d00aa5fd7a15a5109103308` |

## Scenario 2 Retry: Train-Only Template Refresh

| Field | Value |
| --- | --- |
| Train marker | `scenario_2_retry_train_only_template_refresh` |
| Commit | `7c4df03d9c18b889454c4d1849f822d203f08871` |
| Re-registered templates | only Train Model |
| New Train Model template ID | `80a1b704eaf2488b92e5dc25152901d8` |
| Controller ID | `dadb3a8df75f44d58116c81cf30a089d` |
| Controller status | `completed` |

| Step | Baseline task ID | Scenario 2 task ID | Actual result |
| --- | --- | --- | --- |
| extract | `d8871a59ef9f427aadae593ff82ade8d` | `af51faa522fe4df58292d2040ddbcd18` | Executed new task at `7c4df03`; not reused. |
| feature | `a1162a52d6104786a8e685d5add7e1d3` | `e9fa986b2b7a424c9dc94a92ea16645d` | Executed new task at `7c4df03`; not reused. |
| validate | `cf348eeb117d46bcbcbb6974cd524957` | `0266e4d755424ef994f05d7a8e8b4eea` | Executed new task at `7c4df03`; not reused. |
| drift | `f68d01361db74ca489ca55c6e316dfd3` | `d118c2ce58514c89b1803717f230c056` | Executed new task at `7c4df03`; not reused. |
| hpo | `cba14ee087744da787a117d7f9e2826b` | `52af2481d35245da90ae91d8b636eadb` | Executed new task at `7c4df03`; not reused. |
| train | `322fdb5748764be6bf80ee55340c1be5` | `b0a7a01f81d347fd8062aea10ae5da7b` | Executed new task and logged `scenario_2_retry_train_only_template_refresh`. |
| evaluate | `69abe032e2f74a2b98ec1ba1520faa44` | `ff07a0814f4c4211b81b288a0adfb4ca` | Executed new downstream task. |
| register | `8f242e4266af444cabd8565ef8cfd4b9` | `6f77adaec99b4278b2731352b2e65fd7` | Executed new downstream task. |

Actual conclusion: re-registering only Train Model did not cause upstream cache reuse. Even unchanged upstream templates executed as new tasks because the ClearML agent resolved the branch to the new Git commit, and the child task execution commit changed from `0ae9c40` to `7c4df03`.

## Scenario 3 Retry: Commit Without Re-Registering Templates

| Field | Value |
| --- | --- |
| Train marker | `scenario_3_retry_no_template_refresh` |
| Commit | `022dca3818c3973ab69d7e68ef4bf335328ad2a0` |
| Re-registered templates | none |
| Train Model template ID | remained `80a1b704eaf2488b92e5dc25152901d8` |
| Controller ID | `9e161ba063bd4a84b0da866445bf0c64` |
| Controller status | `completed` |

| Step | Scenario 2 task ID | Scenario 3 task ID | Actual result |
| --- | --- | --- | --- |
| extract | `af51faa522fe4df58292d2040ddbcd18` | `f501752d0f8a4595beab6fc046ef4d1c` | Executed new task at `022dca3`. |
| feature | `e9fa986b2b7a424c9dc94a92ea16645d` | `cc033c5dbc9945338a334c9721b01f54` | Executed new task at `022dca3`. |
| validate | `0266e4d755424ef994f05d7a8e8b4eea` | `6d85d24f530b411eae3863cd935d53eb` | Executed new task at `022dca3`. |
| drift | `d118c2ce58514c89b1803717f230c056` | `aa5e52814edd46d8be4017cd4813ef29` | Executed new task at `022dca3`. |
| hpo | `52af2481d35245da90ae91d8b636eadb` | `44e6c5fd1cd24edeaf317df9a7ca378a` | Executed new task at `022dca3`. |
| train | `b0a7a01f81d347fd8062aea10ae5da7b` | `faa1f4f4dee245199bf9a6487554eac9` | Old template ID still ran new branch HEAD and logged `scenario_3_retry_no_template_refresh`. |

Actual conclusion: in this environment, commit/push alone is enough for the existing template to run new code because templates are branch-based, not commit-pinned. Re-registering the template is useful for traceability/template ID records, but it was not required for code execution in this branch-tracking setup.

## Scenario 4 Retry: Failed Train and New Pipeline Recovery

### Failed Pipeline

| Field | Value |
| --- | --- |
| Failed commit | `922d415ef0beeaa7f2514fcd0faa1c552c06c6f9` |
| Failing Train Model template ID | `bd665008567343b59be66c24cddc98b8` |
| Controller ID | `895b7d6b89b64fcca030892f92d43c08` |
| Controller status before manual repair | `failed` |
| Failed train task | `9a946972b9c34844bdabde40fe2c3e1b` |

| Step | Task ID | Status | Commit |
| --- | --- | --- | --- |
| extract | `b951e8fbf5df4dbcb0c028411b883446` | completed | `922d415ef0beeaa7f2514fcd0faa1c552c06c6f9` |
| feature | `a551245b4a5645138e35bcccea2bf653` | completed | `922d415ef0beeaa7f2514fcd0faa1c552c06c6f9` |
| validate | `27c95bade2924eceaffba1dcd6a1e332` | completed | `922d415ef0beeaa7f2514fcd0faa1c552c06c6f9` |
| drift | `40682fbd2ceb451599d0ea6343cee64a` | completed | `922d415ef0beeaa7f2514fcd0faa1c552c06c6f9` |
| hpo | `58aa5c7fa13b4430a5bf4383b367117d` | completed | `922d415ef0beeaa7f2514fcd0faa1c552c06c6f9` |
| train | `9a946972b9c34844bdabde40fe2c3e1b` | failed | `922d415ef0beeaa7f2514fcd0faa1c552c06c6f9` |

Failure evidence:

```text
Cache experiment trace: scenario_4_controlled_train_failure
Controlled cache experiment train failure requested.
RuntimeError: Controlled cache experiment train failure.
Process failed, exit code 1
```

### New Pipeline Recovery

| Field | Value |
| --- | --- |
| Fixed commit | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |
| Fixed Train Model template ID | `49832be9525e4bb98aa642089d2c71d4` |
| Controller ID | `9483e5f99b174068913a630b0e84b736` |
| Controller status | `failed` |
| Reason | Fixed train/evaluate/register succeeded, then the controller failed while launching the all-cached downstream parallel tail. |

| Step | Task ID | Status | Commit |
| --- | --- | --- | --- |
| extract | `bcd90c44ae284a4a88cc2856ac1b6b04` | completed | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |
| feature | `e6eaa8be67f94cc483d9f4438c462f0d` | completed | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |
| validate | `b26ea5f57ff543648021fac75265b044` | completed | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |
| drift | `5e21835cc33e465dabbaa56bea90049a` | completed | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |
| hpo | `a1b7421604e0491385c0f20a3f0570a9` | completed | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |
| train | `7d6676b5dde04e81bce6fde131d79c3a` | completed | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |
| evaluate | `7b2f8f03f777407ebc7a1cc0d25f7acd` | completed | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |
| register | `7d60ca49a5204fd6848dbb11e94ac346` | completed | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |
| deploy_candidate_serving | `390fd4acc760407aab2415d7b0ec589e` | stopped after cleanup | no resolved commit |
| compare_champion | `fdb4b44349704254a11bf93465d8a341` | completed | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |
| explain_model | `1a163e0350d54af9aa2fd217afde5b1f` | completed | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |

Fixed train evidence:

```text
Cache experiment trace: scenario_4_fixed_train_recovery
Process completed successfully
```

Controller failure evidence from local controller output:

```text
ValueError: Task object can only be updated if created or in_progress [status=completed fields=['hyperparams']]
Setting pipeline controller Task as failed (due to failed steps) !
```

Actual conclusion: starting a new pipeline with the fixed commit is the correct way to rerun the failed train task. However, with every downstream training step cached, this repo/server combination exposed a ClearML controller failure in the parallel downstream tail. Upstream successful steps from the failed run were not reused because the fixed commit changed the branch HEAD from `922d415` to `49f642a`.

## Scenario 5 Retry: Same Failed Pipeline Manual Repair

Manual repair action performed after the Scenario 4 failed-pipeline snapshot:

1. Reset failed train child `9a946972b9c34844bdabde40fe2c3e1b`.
2. Set script branch to `test-clearml-cache-behavior`.
3. Set script commit to fixed commit `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e`.
4. Re-enqueue the same child task on queue `training`.

| Field | Value |
| --- | --- |
| Parent pipeline | `895b7d6b89b64fcca030892f92d43c08` |
| Parent status before repair | `failed` |
| Parent status after repaired child completed | `failed` |
| Repaired child | `9a946972b9c34844bdabde40fe2c3e1b` |
| Child status after repair | `completed` |
| Child commit after repair | `49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e` |

Repair evidence:

```text
before parent failed
before train failed test-clearml-cache-behavior 922d415ef0beeaa7f2514fcd0faa1c552c06c6f9
after reset created test-clearml-cache-behavior 49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e
after enqueue queued
parent failed
train completed 49f642a4c15cc173a5b3732fbe2c3ae9539a8f5e
```

Actual conclusion: the child task can be manually repaired and made successful, but the already-failed parent pipeline does not resume and no downstream steps are scheduled. This is not a reliable operational recovery path.

## Continuation Final Conclusion

| Question | Observed answer |
| --- | --- |
| Do unchanged upstream steps cache across train-only commits? | No, not in this setup. Upstream tasks executed new child task IDs at every new branch HEAD commit. |
| Is re-registering all templates required after a train code change? | No for code execution in this branch-based setup. Scenario 3 proved the old train template ran new branch HEAD code after commit/push only. |
| Is re-registering only Train useful? | Yes for explicit template lineage, but it did not force upstream cache reuse. |
| Are failed tasks cached? | The failed train task was not reused as success. It had to rerun after reset or in a new pipeline. |
| Does new-pipeline recovery work? | Train/evaluate/register recovery worked, but all-cached downstream parallel tail exposed a ClearML controller failure. |
| Does manual repair resume the same failed pipeline? | No. The repaired train child completed, but the parent pipeline remained failed. |

Recommended workflow from observed behavior:

- Treat templates as branch-tracking unless you explicitly pin commits in the template task.
- If you need deterministic cache keys, pin template execution to a commit instead of only `branch=test-clearml-cache-behavior`.
- For operational recovery, start a new pipeline after committing and pushing the fix.
- Do not rely on manually resetting a failed child task to resume the original parent pipeline.
- Be cautious with `cache_executed_step=True` on service/deploy/verify and other downstream side-effect steps. The all-cached experiment exposed a ClearML controller failure while launching cached downstream steps with parameter overrides.
- For production, keep deploy/verify/service side-effect steps uncached unless there is a specific, tested reason to cache them.
