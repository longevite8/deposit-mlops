import os
from pathlib import Path
from dotenv import load_dotenv
from clearml import Task
from subprocess import check_output, run
from config import (
    TEMPLATE_EXTRACT_NAME,
    TEMPLATE_FEATURE_NAME,
    TEMPLATE_VALIDATE_NAME,
    TEMPLATE_DRIFT_NAME,
    TEMPLATE_HPO_NAME,
    TEMPLATE_TRAIN_NAME,
    TEMPLATE_EVALUATE_NAME,
    TEMPLATE_REGISTER_NAME,
    TEMPLATE_COMPARE_CHAMPION_NAME,
    TEMPLATE_PROMOTE_CHAMPION_NAME,
    TEMPLATE_INFERENCE_NAME,
    TEMPLATE_MONITORING_NAME,
    TEMPLATE_ALERTING_NAME,
    TEMPLATE_AUTO_RETRAINING_NAME,
    TEMPLATE_DEPLOY_SERVING_NAME,
    TEMPLATE_EXPLAIN_NAME,
    TEMPLATE_DEPLOY_CANDIDATE_SERVING_NAME,
    TEMPLATE_VERIFY_ENDPOINT_NAME,
    TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_NAME,
    PROJECT_TEMPLATE,
)

# =====================================================
# Load environment variables
# =====================================================

# Load .env từ project root
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# Load GIT_REPO từ .env hoặc dùng default
GIT_REPO = os.getenv("GIT_REPO", "")
GIT_BRANCH = os.getenv("GIT_BRANCH", "vc-mco")
REQUIREMENTS_FILE = os.getenv("CLEARML_REQUIREMENTS_FILE", "requirements.txt")
LIGHT_REQUIREMENTS_FILE = os.getenv(
    "CLEARML_LIGHT_REQUIREMENTS_FILE", "requirements-tasks.txt"
)
FORECAST_REQUIREMENTS_FILE = os.getenv(
    "CLEARML_FORECAST_REQUIREMENTS_FILE", "requirements-forecast.txt"
)

if not GIT_REPO:
    raise ValueError(
        "❌ GIT_REPO not set in .env file\n"
        "   Add line: GIT_REPO=https://github.com/your-username/cashflow-clearml.git"
    )

print(f"✅ Using GIT_REPO: {GIT_REPO}")
print(f"✅ Using GIT_BRANCH: {GIT_BRANCH}")
print(f"✅ Using default requirements file: {REQUIREMENTS_FILE}")
print(f"✅ Using light task requirements file: {LIGHT_REQUIREMENTS_FILE}")
print(f"✅ Using forecast requirements file: {FORECAST_REQUIREMENTS_FILE}")


def update_env_file(env_file: Path, values: dict[str, str]) -> None:
    """Upsert generated ClearML IDs into the local ignored .env file."""

    existing_lines = []
    if env_file.exists():
        existing_lines = env_file.read_text(encoding="utf-8").splitlines()

    remaining = dict(values)
    updated_lines = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue

        key = line.split("=", 1)[0].strip()
        if key in remaining:
            updated_lines.append(f"{key}={remaining.pop(key)}")
        else:
            updated_lines.append(line)

    if remaining and updated_lines and updated_lines[-1].strip():
        updated_lines.append("")
    for key, value in remaining.items():
        updated_lines.append(f"{key}={value}")

    env_file.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

# Mảng mapping giữa Tên Template, Loại Task, Script, Tên biến trong config.py,
# và requirements profile dành cho ClearML Agent.
templates = [
    (
        TEMPLATE_EXTRACT_NAME,
        Task.TaskTypes.data_processing,
        "tasks/extract_data.py",
        "TEMPLATE_EXTRACT_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_FEATURE_NAME,
        Task.TaskTypes.data_processing,
        "tasks/feature_engineering.py",
        "TEMPLATE_FEATURE_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_VALIDATE_NAME,
        Task.TaskTypes.qc,
        "tasks/validate_data.py",
        "TEMPLATE_VALIDATE_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_DRIFT_NAME,
        Task.TaskTypes.qc,
        "tasks/drift_detection.py",
        "TEMPLATE_DRIFT_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_HPO_NAME,
        Task.TaskTypes.optimizer,
        "tasks/hpo_model.py",
        "TEMPLATE_HPO_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_TRAIN_NAME,
        Task.TaskTypes.training,
        "tasks/train_model.py",
        "TEMPLATE_TRAIN_ID",
        FORECAST_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_EVALUATE_NAME,
        Task.TaskTypes.qc,
        "tasks/evaluate_model.py",
        "TEMPLATE_EVALUATE_ID",
        FORECAST_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_REGISTER_NAME,
        Task.TaskTypes.application,
        "tasks/register_model.py",
        "TEMPLATE_REGISTER_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_COMPARE_CHAMPION_NAME,
        Task.TaskTypes.application,
        "tasks/compare_champion.py",
        "TEMPLATE_COMPARE_CHAMPION_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_PROMOTE_CHAMPION_NAME,
        Task.TaskTypes.application,
        "tasks/promote_champion.py",
        "TEMPLATE_PROMOTE_CHAMPION_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_INFERENCE_NAME,
        Task.TaskTypes.inference,
        "tasks/inference_model.py",
        "TEMPLATE_INFERENCE_ID",
        FORECAST_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_MONITORING_NAME,
        Task.TaskTypes.qc,
        "tasks/monitoring_model.py",
        "TEMPLATE_MONITORING_ID",
        FORECAST_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_ALERTING_NAME,
        Task.TaskTypes.qc,
        "tasks/alerting_model.py",
        "TEMPLATE_ALERTING_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_AUTO_RETRAINING_NAME,
        Task.TaskTypes.application,
        "tasks/auto_retraining.py",
        "TEMPLATE_AUTO_RETRAINING_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_EXPLAIN_NAME,
        Task.TaskTypes.qc,
        "tasks/explain_model.py",
        "TEMPLATE_EXPLAIN_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_DEPLOY_SERVING_NAME,
        Task.TaskTypes.service,
        "tasks/deploy_serving.py",
        "TEMPLATE_DEPLOY_SERVING_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_DEPLOY_CANDIDATE_SERVING_NAME,
        Task.TaskTypes.service,
        "tasks/deploy_candidate_serving.py",
        "TEMPLATE_DEPLOY_CANDIDATE_SERVING_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_VERIFY_ENDPOINT_NAME,
        Task.TaskTypes.qc,
        "tasks/verify_endpoint.py",
        "TEMPLATE_VERIFY_ENDPOINT_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
    (
        TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_NAME,
        Task.TaskTypes.qc,
        "tasks/verify_candidate_endpoint.py",
        "TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_ID",
        LIGHT_REQUIREMENTS_FILE,
    ),
]

def current_worktree_diff(paths: list[str] | None = None) -> str:
    """Return a git patch for tracked and untracked local changes."""

    diff_command = ["git", "diff", "--no-ext-diff", "--unified=0"]
    if paths:
        diff_command.extend(["--", *paths])
    tracked_diff = check_output(diff_command).decode()
    untracked = check_output(
        ["git", "ls-files", "--others", "--exclude-standard"]
    ).decode().splitlines()
    if paths:
        path_set = set(paths)
        untracked = [path for path in untracked if path in path_set]

    untracked_diffs = []
    for path in untracked:
        result = run(
            ["git", "diff", "--no-index", "--", "/dev/null", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            untracked_diffs.append(result.stdout)

    return "\n".join([tracked_diff, *untracked_diffs]).strip()


current_commit = check_output(["git", "rev-parse", "HEAD"]).decode().strip()
include_worktree_diff = os.getenv("CLEARML_INCLUDE_WORKTREE_DIFF", "false").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
if include_worktree_diff:
    print("⚠️ Including local uncommitted script diffs in template tasks")
else:
    print("✅ Registering templates from Git commit only (no worktree diff)")

# Dictionary để lưu ID mới nhằm cập nhật vào config.py
new_ids = {}

for name, task_type, script, config_var, requirements_file in templates:
    task = Task.create(
        project_name=PROJECT_TEMPLATE,
        task_name=name,
        task_type=task_type,
        repo=GIT_REPO,
        branch=GIT_BRANCH,
        script=script,
        working_directory=".",
        requirements_file=requirements_file,
    )
    script_diff = current_worktree_diff([script]) if include_worktree_diff else ""
    task.set_script(diff=script_diff)

    task.upload_artifact("template_commit", current_commit)
    new_ids[config_var] = task.id
    print(f"✅ Created {name}: {task.id}")

# =====================================================
# Tự động cập nhật .env cục bộ (không commit ClearML IDs mới vào code)
# =====================================================

update_env_file(env_path, new_ids)
print(f"\n🚀 Successfully updated {len(new_ids)} template IDs in {env_path}")
print("   Keep .env local; do not commit generated ClearML resource IDs.")
