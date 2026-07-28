import argparse
import os
from pathlib import Path
import re
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
# Registration settings
# =====================================================

ENV_PATH = Path(__file__).parent / ".env"


def load_registration_settings(env_path: Path = ENV_PATH) -> dict[str, str]:
    """Load ClearML template registration settings from the project environment."""

    load_dotenv(env_path)
    settings = {
        "GIT_REPO": os.getenv("GIT_REPO", ""),
        "GIT_BRANCH": os.getenv("GIT_BRANCH", "vc-mco"),
        "REQUIREMENTS_FILE": os.getenv("CLEARML_REQUIREMENTS_FILE", "requirements.txt"),
        "LIGHT_REQUIREMENTS_FILE": os.getenv(
            "CLEARML_LIGHT_REQUIREMENTS_FILE", "requirements-tasks.txt"
        ),
        "FORECAST_REQUIREMENTS_FILE": os.getenv(
            "CLEARML_FORECAST_REQUIREMENTS_FILE", "requirements-forecast.txt"
        ),
    }

    if not settings["GIT_REPO"]:
        raise ValueError(
            "❌ GIT_REPO not set in .env file\n"
            "   Add line: GIT_REPO=https://github.com/your-username/cashflow-clearml.git"
        )

    return settings


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

def build_templates(
    light_requirements_file: str,
    forecast_requirements_file: str,
) -> list[tuple[str, str, str, str, str]]:
    """Build the template registration matrix."""

    # Mảng mapping giữa Tên Template, Loại Task, Script, Tên biến trong config.py,
    # và requirements profile dành cho ClearML Agent.
    return [
        (
            TEMPLATE_EXTRACT_NAME,
            Task.TaskTypes.data_processing,
            "tasks/extract_data.py",
            "TEMPLATE_EXTRACT_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_FEATURE_NAME,
            Task.TaskTypes.data_processing,
            "tasks/feature_engineering.py",
            "TEMPLATE_FEATURE_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_VALIDATE_NAME,
            Task.TaskTypes.qc,
            "tasks/validate_data.py",
            "TEMPLATE_VALIDATE_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_DRIFT_NAME,
            Task.TaskTypes.qc,
            "tasks/drift_detection.py",
            "TEMPLATE_DRIFT_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_HPO_NAME,
            Task.TaskTypes.optimizer,
            "tasks/hpo_model.py",
            "TEMPLATE_HPO_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_TRAIN_NAME,
            Task.TaskTypes.training,
            "tasks/train_model.py",
            "TEMPLATE_TRAIN_ID",
            forecast_requirements_file,
        ),
        (
            TEMPLATE_EVALUATE_NAME,
            Task.TaskTypes.qc,
            "tasks/evaluate_model.py",
            "TEMPLATE_EVALUATE_ID",
            forecast_requirements_file,
        ),
        (
            TEMPLATE_REGISTER_NAME,
            Task.TaskTypes.application,
            "tasks/register_model.py",
            "TEMPLATE_REGISTER_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_COMPARE_CHAMPION_NAME,
            Task.TaskTypes.application,
            "tasks/compare_champion.py",
            "TEMPLATE_COMPARE_CHAMPION_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_PROMOTE_CHAMPION_NAME,
            Task.TaskTypes.application,
            "tasks/promote_champion.py",
            "TEMPLATE_PROMOTE_CHAMPION_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_INFERENCE_NAME,
            Task.TaskTypes.inference,
            "tasks/inference_model.py",
            "TEMPLATE_INFERENCE_ID",
            forecast_requirements_file,
        ),
        (
            TEMPLATE_MONITORING_NAME,
            Task.TaskTypes.qc,
            "tasks/monitoring_model.py",
            "TEMPLATE_MONITORING_ID",
            forecast_requirements_file,
        ),
        (
            TEMPLATE_ALERTING_NAME,
            Task.TaskTypes.qc,
            "tasks/alerting_model.py",
            "TEMPLATE_ALERTING_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_AUTO_RETRAINING_NAME,
            Task.TaskTypes.application,
            "tasks/auto_retraining.py",
            "TEMPLATE_AUTO_RETRAINING_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_EXPLAIN_NAME,
            Task.TaskTypes.qc,
            "tasks/explain_model.py",
            "TEMPLATE_EXPLAIN_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_DEPLOY_SERVING_NAME,
            Task.TaskTypes.service,
            "tasks/deploy_serving.py",
            "TEMPLATE_DEPLOY_SERVING_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_DEPLOY_CANDIDATE_SERVING_NAME,
            Task.TaskTypes.service,
            "tasks/deploy_candidate_serving.py",
            "TEMPLATE_DEPLOY_CANDIDATE_SERVING_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_VERIFY_ENDPOINT_NAME,
            Task.TaskTypes.qc,
            "tasks/verify_endpoint.py",
            "TEMPLATE_VERIFY_ENDPOINT_ID",
            light_requirements_file,
        ),
        (
            TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_NAME,
            Task.TaskTypes.qc,
            "tasks/verify_candidate_endpoint.py",
            "TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_ID",
            light_requirements_file,
        ),
    ]


def normalize_selector(value: str) -> str:
    """Normalize a human template selector such as train_model.py or TEMPLATE_TRAIN_ID."""

    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def split_selector_values(values: list[str]) -> list[str]:
    """Split repeated/comma-separated selector values."""

    selectors: list[str] = []
    for value in values:
        selectors.extend(part.strip() for part in value.split(",") if part.strip())
    return selectors


def template_aliases(template: tuple[str, str, str, str, str]) -> set[str]:
    """Return accepted selector aliases for a template tuple."""

    name, _task_type, script, config_var, _requirements_file = template
    script_path = Path(script)
    config_short = config_var
    if config_short.startswith("TEMPLATE_"):
        config_short = config_short[len("TEMPLATE_") :]
    if config_short.endswith("_ID"):
        config_short = config_short[: -len("_ID")]

    aliases = {
        name,
        script,
        script_path.name,
        script_path.stem,
        config_var,
        config_short,
    }
    return {normalize_selector(alias) for alias in aliases}


def select_templates(
    templates: list[tuple[str, str, str, str, str]],
    selectors: list[str],
) -> list[tuple[str, str, str, str, str]]:
    """Filter templates by friendly selectors, preserving registration order."""

    if not selectors:
        return templates

    normalized_selectors = {normalize_selector(selector) for selector in selectors}
    selected = [
        template
        for template in templates
        if template_aliases(template) & normalized_selectors
    ]
    matched_aliases = set().union(*(template_aliases(template) for template in selected))
    unknown = sorted(normalized_selectors - matched_aliases)
    if unknown:
        accepted = sorted(
            {
                alias
                for template in templates
                for alias in template_aliases(template)
                if alias
            }
        )
        raise ValueError(
            "Unknown template selector(s): "
            f"{', '.join(unknown)}. Accepted examples: {', '.join(accepted)}"
        )

    return selected


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register ClearML task templates and update local .env IDs."
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help=(
            "Register only matching template(s). Accepts comma-separated aliases "
            "such as train, TEMPLATE_TRAIN_ID, Train Model, or tasks/train_model.py. "
            "Can also be supplied with REGISTER_TEMPLATE_ONLY."
        ),
    )
    return parser


def env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "y"}


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    settings = load_registration_settings()
    templates = build_templates(
        light_requirements_file=settings["LIGHT_REQUIREMENTS_FILE"],
        forecast_requirements_file=settings["FORECAST_REQUIREMENTS_FILE"],
    )
    only_values = split_selector_values(
        [*args.only, os.getenv("REGISTER_TEMPLATE_ONLY", "")]
    )
    selected_templates = select_templates(templates, only_values)

    print(f"✅ Using GIT_REPO: {settings['GIT_REPO']}")
    print(f"✅ Using GIT_BRANCH: {settings['GIT_BRANCH']}")
    print(f"✅ Using default requirements file: {settings['REQUIREMENTS_FILE']}")
    print(
        f"✅ Using light task requirements file: {settings['LIGHT_REQUIREMENTS_FILE']}"
    )
    print(
        f"✅ Using forecast requirements file: {settings['FORECAST_REQUIREMENTS_FILE']}"
    )
    if only_values:
        selected_names = ", ".join(template[0] for template in selected_templates)
        print(f"✅ Registering selected template(s): {selected_names}")
    else:
        print("✅ Registering all templates")

    current_commit = check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    include_worktree_diff = env_flag("CLEARML_INCLUDE_WORKTREE_DIFF")
    if include_worktree_diff:
        print("⚠️ Including local uncommitted script diffs in template tasks")
    else:
        print("✅ Registering templates from Git commit only (no worktree diff)")

    # Dictionary để lưu ID mới nhằm cập nhật vào config.py
    new_ids = {}

    for name, task_type, script, config_var, requirements_file in selected_templates:
        task = Task.create(
            project_name=PROJECT_TEMPLATE,
            task_name=name,
            task_type=task_type,
            repo=settings["GIT_REPO"],
            branch=settings["GIT_BRANCH"],
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

    update_env_file(ENV_PATH, new_ids)
    print(f"\n🚀 Successfully updated {len(new_ids)} template IDs in {ENV_PATH}")
    print("   Keep .env local; do not commit generated ClearML resource IDs.")


if __name__ == "__main__":
    main()
