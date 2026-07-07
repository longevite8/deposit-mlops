import os
import re
from pathlib import Path
from dotenv import load_dotenv
from clearml import Task
from subprocess import check_output
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
    TEMPLATE_EXPLAIN_NAME,
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

if not GIT_REPO:
    raise ValueError(
        "❌ GIT_REPO not set in .env file\n"
        "   Add line: GIT_REPO=https://github.com/your-username/cashflow-clearml.git"
    )

print(f"✅ Using GIT_REPO: {GIT_REPO}")

# Mảng mapping giữa Tên Template, Loại Task, Script và Tên biến trong config.py
templates = [
    (
        TEMPLATE_EXTRACT_NAME,
        Task.TaskTypes.data_processing,
        "tasks/extract_data.py",
        "TEMPLATE_EXTRACT_ID",
    ),
    (
        TEMPLATE_FEATURE_NAME,
        Task.TaskTypes.data_processing,
        "tasks/feature_engineering.py",
        "TEMPLATE_FEATURE_ID",
    ),
    (
        TEMPLATE_VALIDATE_NAME,
        Task.TaskTypes.qc,
        "tasks/validate_data.py",
        "TEMPLATE_VALIDATE_ID",
    ),
    (
        TEMPLATE_DRIFT_NAME,
        Task.TaskTypes.qc,
        "tasks/drift_detection.py",
        "TEMPLATE_DRIFT_ID",
    ),
    (
        TEMPLATE_HPO_NAME,
        Task.TaskTypes.optimizer,
        "tasks/hpo_model.py",
        "TEMPLATE_HPO_ID",
    ),
    (
        TEMPLATE_TRAIN_NAME,
        Task.TaskTypes.training,
        "tasks/train_model.py",
        "TEMPLATE_TRAIN_ID",
    ),
    (
        TEMPLATE_EVALUATE_NAME,
        Task.TaskTypes.qc,
        "tasks/evaluate_model.py",
        "TEMPLATE_EVALUATE_ID",
    ),
    (
        TEMPLATE_REGISTER_NAME,
        Task.TaskTypes.application,
        "tasks/register_model.py",
        "TEMPLATE_REGISTER_ID",
    ),
    (
        TEMPLATE_COMPARE_CHAMPION_NAME,
        Task.TaskTypes.application,
        "tasks/compare_champion.py",
        "TEMPLATE_COMPARE_CHAMPION_ID",
    ),
    (
        TEMPLATE_PROMOTE_CHAMPION_NAME,
        Task.TaskTypes.application,
        "tasks/promote_champion.py",
        "TEMPLATE_PROMOTE_CHAMPION_ID",
    ),
    (
        TEMPLATE_INFERENCE_NAME,
        Task.TaskTypes.inference,
        "tasks/inference_model.py",
        "TEMPLATE_INFERENCE_ID",
    ),
    (
        TEMPLATE_MONITORING_NAME,
        Task.TaskTypes.qc,
        "tasks/monitoring_model.py",
        "TEMPLATE_MONITORING_ID",
    ),
    (
        TEMPLATE_ALERTING_NAME,
        Task.TaskTypes.qc,
        "tasks/alerting_model.py",
        "TEMPLATE_ALERTING_ID",
    ),
    (
        TEMPLATE_AUTO_RETRAINING_NAME,
        Task.TaskTypes.application,
        "tasks/auto_retraining.py",
        "TEMPLATE_AUTO_RETRAINING_ID",
    ),
    (
        TEMPLATE_EXPLAIN_NAME,
        Task.TaskTypes.qc,
        "tasks/explain_model.py",
        "TEMPLATE_EXPLAIN_ID",
    ),
]

current_commit = check_output(["git", "rev-parse", "HEAD"]).decode().strip()

# Dictionary để lưu ID mới nhằm cập nhật vào config.py
new_ids = {}

for name, task_type, script, config_var in templates:
    task = Task.create(
        project_name=PROJECT_TEMPLATE,
        task_name=name,
        task_type=task_type,
        repo=GIT_REPO,
        branch="main",
        script=script,
        working_directory=".",
    )

    task.upload_artifact("template_commit", current_commit)
    new_ids[config_var] = task.id
    print(f"✅ Created {name}: {task.id}")

# =====================================================
# Tự động cập nhật config.py
# =====================================================

config_path = Path(__file__).parent / "config.py"
config_content = config_path.read_text(encoding="utf-8")

for var_name, new_id in new_ids.items():
    # Sử dụng Regex để tìm biến và thay thế giá trị trong dấu ngoặc kép
    pattern = rf'({var_name}\s*=\s*")[^"]*(")'
    # SỬA: Sử dụng \g<1> và \g<2> để tránh xung đột khi new_id bắt đầu bằng chữ số
    replacement = rf"\g<1>{new_id}\g<2>"
    config_content = re.sub(pattern, replacement, config_content)

config_path.write_text(config_content, encoding="utf-8")
print(f"\n🚀 Successfully updated {len(new_ids)} IDs in config.py")
