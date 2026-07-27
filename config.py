"""
Central configuration file for Cashflow ClearML MLOps pipeline.
All constants should be defined here — never hardcode values in task files.
"""

import os

from dotenv import load_dotenv

# =====================================================
# Load environment variables từ .env file
# =====================================================

load_dotenv()

# =====================================================
# Project & Environment
# =====================================================

PROJECT_PARENT = "Deposit-CashFlow"
PROJECT_TEMPLATE = f"{PROJECT_PARENT}/Templates"
PROJECT_PIPELINE = f"{PROJECT_PARENT}/Pipelines"
PROJECT_DATASET = f"{PROJECT_PARENT}/Datasets"
TRAINING_PIPELINE_NAME = "Training Pipeline"
PRODUCTION_PIPELINE_NAME = "Production Pipeline"

DEPLOYMENT_VERSION = "1.0.0"  # Versioning for production deployment
RANDOM_STATE = 42

# =====================================================
# THÊM: ClearML Server Configuration (từ Environment Variable)
# =====================================================

CLEARML_SERVER_URL = os.getenv(
    "CLEARML_SERVER_URL",
    "http://192.168.140.248:8080",  # ← Default value nếu env var không set
)

# =====================================================
# Queues
# =====================================================

CPU_QUEUE = os.getenv("CLEARML_CPU_QUEUE", "training")
SERVICES_QUEUE = os.getenv("CLEARML_SERVICES_QUEUE", "mco-services")
RUN_PIPELINE_CONTROLLER_LOCALLY = os.getenv(
    "RUN_PIPELINE_CONTROLLER_LOCALLY", "true"
).lower() in {"1", "true", "yes", "y"}

# =====================================================
# Production Pipeline Scheduler
# =====================================================

PRODUCTION_PIPELINE_ID = os.getenv("PRODUCTION_PIPELINE_ID", "")
PRODUCTION_SCHEDULER_PROJECT = os.getenv(
    "PRODUCTION_SCHEDULER_PROJECT",
    PROJECT_PIPELINE,
)
PRODUCTION_SCHEDULER_TASK_NAME = os.getenv(
    "PRODUCTION_SCHEDULER_TASK_NAME",
    "Scheduled Production Pipeline Controller",
)
PRODUCTION_SCHEDULER_JOB_NAME = os.getenv(
    "PRODUCTION_SCHEDULER_JOB_NAME",
    "daily_production_pipeline",
)
PRODUCTION_SCHEDULER_QUEUE = os.getenv(
    "PRODUCTION_SCHEDULER_QUEUE",
    SERVICES_QUEUE,
)
PRODUCTION_SCHEDULER_TARGET_PROJECT = os.getenv(
    "PRODUCTION_SCHEDULER_TARGET_PROJECT",
    PROJECT_PIPELINE,
)
PRODUCTION_SCHEDULER_SYNC_MINUTES = float(
    os.getenv("PRODUCTION_SCHEDULER_SYNC_MINUTES", "5")
)
PRODUCTION_SCHEDULER_START_REMOTELY = os.getenv(
    "PRODUCTION_SCHEDULER_START_REMOTELY",
    "true",
).lower() in {"1", "true", "yes", "y"}
PRODUCTION_SCHEDULER_EXECUTE_IMMEDIATELY = os.getenv(
    "PRODUCTION_SCHEDULER_EXECUTE_IMMEDIATELY",
    "false",
).lower() in {"1", "true", "yes", "y"}
PRODUCTION_SCHEDULER_DAILY_UTC_HOUR = int(
    os.getenv("PRODUCTION_SCHEDULER_DAILY_UTC_HOUR", "19")
)
PRODUCTION_SCHEDULER_DAILY_UTC_MINUTE = int(
    os.getenv("PRODUCTION_SCHEDULER_DAILY_UTC_MINUTE", "0")
)

# =====================================================
# Source Database Configuration
# =====================================================

DB_USER = os.getenv("DB_USER", "vega")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "vc_mco_dev")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")

SOURCE_PROJECT_NAME = os.getenv("SOURCE_PROJECT_NAME", "Tin Vay")
SOURCE_FLOW_TYPE = os.getenv("SOURCE_FLOW_TYPE", "Vào")
SOURCE_APPROVAL_STATUS = os.getenv("SOURCE_APPROVAL_STATUS", "Đã duyệt")
SOURCE_FROM_DATE = os.getenv("SOURCE_FROM_DATE", "2025-01-01")
SOURCE_TO_DATE = os.getenv("SOURCE_TO_DATE", "2026-01-01")

# =====================================================
# Data Validation Thresholds
# =====================================================

MIN_VALUE = float(os.getenv("VALIDATION_MIN_VALUE", "0.0"))
MAX_VALUE = float(os.getenv("VALIDATION_MAX_VALUE", "200000000000.0"))

# Data quality checks
ALLOW_DUPLICATES = False  # Không cho phép duplicate timestamps
MIN_ROWS_REQUIRED = 50  # Tối thiểu 50 rows để train

# =====================================================
# Feature Engineering Configuration
# =====================================================

FEATURE_COLUMNS = ["lag_1", "lag_7", "rolling_mean_7"]
TARGET_COLUMN = "cashflow"  # ← Dùng cho cả raw & processed data

DATE_COLUMN = "date"

# ✅ Lag features (list of integers)
LAG_FEATURES = [1, 7]

# ✅ Rolling features (list of tuples: (feature_name, window))
ROLLING_FEATURES = [("rolling_mean", 7)]  # Có thể thêm ("rolling_std", 7) nếu cần

# Train / Validation / Test split ratios
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2

# =====================================================
# Synthetic Data Generation (extract_data.py)
# =====================================================

START_DATE = "2022-01-01"  # Ngày bắt đầu tạo synthetic data
N_DAYS = 730  # 2 năm dữ liệu hàng ngày

# Gamma distribution parameters (tạo daily deposits)
GAMMA_SHAPE = 2.0  # Shape parameter
GAMMA_SCALE = 500.0  # Scale parameter (mean ≈ shape * scale)

SYNTHETIC_SEED = 42
BASE_CASHFLOW = 1000  # Base daily cashflow amount

# ✅ THÊM: Noise & Trend
NOISE_SCALE = 100.0  # Gaussian noise scale cho synthetic data
TREND_SLOPE = 1.0  # Trend slope (increase/decrease over time)

# =====================================================
# HPO (Hyperparameter Optimization)
# =====================================================

N_TRIALS = 50  # Number of Optuna trials

# LightGBM hyperparameter search space
HPO_LEARNING_RATE_MIN = 0.01
HPO_LEARNING_RATE_MAX = 0.1
HPO_NUM_LEAVES_MIN = 15
HPO_NUM_LEAVES_MAX = 100
HPO_N_ESTIMATORS_MIN = 100
HPO_N_ESTIMATORS_MAX = 500
HPO_MAX_DEPTH_MIN = 5
HPO_MAX_DEPTH_MAX = 15

LGBM_VERBOSE = -1
LGBM_RANDOM_STATE = 42
LGBM_METRIC = "mape"  # Mean Absolute Percentage Error

# =====================================================
# Model Training
# =====================================================

EARLY_STOPPING_ROUNDS = 50
VALIDATION_SPLIT = 0.2

# Forecast model training (NeuralForecast)
def parse_int_list(value: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise ValueError(f"Expected positive integer in list, got {parsed}")
        if parsed not in values:
            values.append(parsed)
    return tuple(values or default)


FORECAST_UNIQUE_ID = os.getenv("FORECAST_UNIQUE_ID", "Deposit_Portfolio")
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "30"))
FORECAST_HORIZONS = parse_int_list(
    os.getenv("FORECAST_HORIZONS", "7,14,30"),
    default=(7, 14, 30),
)
FORECAST_EVAL_HORIZON = int(os.getenv("FORECAST_EVAL_HORIZON", "30"))
FORECAST_HORIZON_METADATA_KEY = "forecast_horizon"


def forecast_horizon_tag(horizon: int | str) -> str:
    return f"horizon:{int(horizon)}"


def forecast_horizon_endpoint_suffix(horizon: int | str) -> str:
    return f"h{int(horizon)}"
FORECAST_INPUT_SIZE = int(os.getenv("FORECAST_INPUT_SIZE", "90"))
FORECAST_LEARNING_RATE = float(os.getenv("FORECAST_LEARNING_RATE", "0.0005"))
FORECAST_MAX_STEPS = int(os.getenv("FORECAST_MAX_STEPS", "100"))
FORECAST_LOSS = os.getenv("FORECAST_LOSS", "MAPE")
FORECAST_OPTIMIZER = os.getenv("FORECAST_OPTIMIZER", "Adam")
FORECAST_ACTIVATION = os.getenv("FORECAST_ACTIVATION", "ReLU")
FORECAST_CV_WINDOWS = int(os.getenv("FORECAST_CV_WINDOWS", "2"))
FORECAST_START_PADDING_ENABLED = os.getenv(
    "FORECAST_START_PADDING_ENABLED", "true"
).lower() in {"1", "true", "yes", "y"}
FORECAST_OUTPUT_DIR = os.getenv("FORECAST_OUTPUT_DIR", "results")

# =====================================================
# Quality Gate Thresholds (Evaluation)
# =====================================================

MAPE_THRESHOLD = 1.2  # 20% — acceptable MAPE
R2_THRESHOLD = -0.1  # R2 score — lower bound

RMSE_THRESHOLD = 500.0  # Root Mean Squared Error threshold
MAE_THRESHOLD = 300.0  # Mean Absolute Error threshold

# =====================================================
# Drift Detection
# =====================================================

DRIFT_PVALUE_THRESHOLD = 0.05
DRIFT_RATIO_WARNING_THRESHOLD = 0.1  # THÊM: Ngưỡng để chuyển từ PASS sang WARNING
DRIFT_RATIO_THRESHOLD = 0.3  # Đây là ngưỡng FAIL (giữ nguyên)
DRIFT_STATUS_LEVELS = ["PASS", "WARNING", "FAIL"]

# =====================================================
# Monitoring Thresholds (Production)
# =====================================================

MONITORING_MAPE_THRESHOLD = 0.25  # Trigger retraining if MAPE > 25%
MONITORING_R2_THRESHOLD = 0.0  # Trigger retraining if R2 < 0
MONITORING_DRIFT_THRESHOLD = "WARNING"  # Trigger if drift_status >= WARNING

MONITORING_WINDOW_SIZE = 100  # Số samples để monitor

# =====================================================
# SHAP Model Explainability
# =====================================================

N_SHAP_SAMPLES = 100  # Number of samples for SHAP analysis (for speed)
SHAP_MAX_DISPLAY = 10  # Top N features to display

# ======================
# Data validation
# ======================

MAX_MISSING_RATE = 0.05
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# =====================================================
# Email Alerting Configuration
# =====================================================

ALERT_SMTP_HOST = "smtp.gmail.com"
ALERT_SMTP_PORT = 587
ALERT_SMTP_USER = "longevite8@gmail.com"
ALERT_SMTP_PASSWORD = os.environ.get("CLEARML_ALERT_SMTP_PASSWORD", "")
ALERT_EMAIL_FROM = "longevite8@gmail.com"
ALERT_EMAIL_TO = ["thonq@vega.com.vn"]
ALERT_EMAIL_SUBJECT_PREFIX = "[CashFlow MLOps Alert]"
ALERT_TEMPLATE_MAPE = "⚠️ Model MAPE exceeded threshold: {mape:.4f} > {threshold:.4f}"
ALERT_TEMPLATE_DRIFT = (
    "⚠️ Data drift detected: {drift_status} (ratio: {drift_ratio:.2%})"
)
ALERT_TEMPLATE_RETRAINING = "🔄 Auto-retraining triggered due to {reason}"

# =====================================================
# Template Task Names
# =====================================================

TEMPLATE_EXTRACT_NAME = "Extract Data"
TEMPLATE_FEATURE_NAME = "Feature Engineering"
TEMPLATE_VALIDATE_NAME = "Validate Data"
TEMPLATE_DRIFT_NAME = "Drift Detection"
TEMPLATE_HPO_NAME = "HPO Model"
TEMPLATE_TRAIN_NAME = "Train Model"
TEMPLATE_EVALUATE_NAME = "Evaluate Model"
TEMPLATE_REGISTER_NAME = "Register Model"
TEMPLATE_COMPARE_CHAMPION_NAME = "Compare Champion"
TEMPLATE_PROMOTE_CHAMPION_NAME = "Promote Champion"
TEMPLATE_INFERENCE_NAME = "Inference Model"
TEMPLATE_MONITORING_NAME = "Monitoring Model"
TEMPLATE_ALERTING_NAME = "Alerting Model"
TEMPLATE_AUTO_RETRAINING_NAME = "Auto Retraining"
TEMPLATE_EXPLAIN_NAME = "Explain Model"
TEMPLATE_DEPLOY_SERVING_NAME = "Deploy Serving"
TEMPLATE_VERIFY_ENDPOINT_NAME = "Verify Endpoint"
TEMPLATE_DEPLOY_CANDIDATE_SERVING_NAME = "Deploy Candidate Serving"
TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_NAME = "Verify Candidate Endpoint"

# =====================================================
# ClearML Serving
# =====================================================

CLEARML_SERVING_SERVICE_ID = os.getenv("CLEARML_SERVING_SERVICE_ID", "")
CLEARML_SERVING_SERVICE_NAME = os.getenv(
    "CLEARML_SERVING_SERVICE_NAME",
    "Deposit CashFlow Serving",
)
CLEARML_SERVING_PROJECT = os.getenv(
    "CLEARML_SERVING_PROJECT",
    f"{PROJECT_PARENT}/Serving",
)
CLEARML_SERVING_ENDPOINT_PREFIX = os.getenv(
    "CLEARML_SERVING_ENDPOINT_PREFIX",
    "deposit/cashflow",
)
CLEARML_SERVING_ENDPOINT = os.getenv("CLEARML_SERVING_ENDPOINT", "")
CLEARML_SERVING_ENDPOINT_VERSION = os.getenv("CLEARML_SERVING_ENDPOINT_VERSION", "")
CLEARML_SERVING_PREPROCESS = os.getenv(
    "CLEARML_SERVING_PREPROCESS",
    "serving/clearml_preprocess.py",
)
CLEARML_SERVING_ENGINE = os.getenv("CLEARML_SERVING_ENGINE", "sklearn")
CLEARML_SERVING_ALIAS = os.getenv("CLEARML_SERVING_ALIAS", "champion")
CLEARML_CANDIDATE_SERVING_ALIAS = os.getenv(
    "CLEARML_CANDIDATE_SERVING_ALIAS", "candidate"
)
CLEARML_CANDIDATE_SERVING_ENDPOINT = os.getenv(
    "CLEARML_CANDIDATE_SERVING_ENDPOINT", "deposit/cashflow/candidate"
)
CLEARML_CANDIDATE_SERVING_ENDPOINT_VERSION = os.getenv(
    "CLEARML_CANDIDATE_SERVING_ENDPOINT_VERSION", ""
)
CLEARML_SERVING_BASE_URL = os.getenv(
    "CLEARML_SERVING_BASE_URL",
    os.getenv("CLEARML_SERVING_INFERENCE_URL", "http://127.0.0.1:8082"),
)
CLEARML_BASE_SERVING_URL = os.getenv(
    "CLEARML_BASE_SERVING_URL",
    os.getenv("CLEARML_DEFAULT_BASE_SERVE_URL", ""),
)
CLEARML_SERVING_METRIC_LOG_FREQ = float(
    os.getenv("CLEARML_SERVING_METRIC_LOG_FREQ", "1.0")
)
AUTO_CREATE_CLEARML_SERVING_SERVICE = os.getenv(
    "AUTO_CREATE_CLEARML_SERVING_SERVICE",
    "0",
).lower() in {"1", "true", "yes", "y"}

# =====================================================
# Template Task IDs (Populated by register_templates.py)
# =====================================================


def env_template_id(name, default):
    """Read a ClearML template ID from the environment with legacy fallback."""

    return os.getenv(name, default)


TEMPLATE_EXTRACT_ID = env_template_id(
    "TEMPLATE_EXTRACT_ID", "7034b2119ab946a1a8aaf01918dfedaf"
)
TEMPLATE_FEATURE_ID = env_template_id(
    "TEMPLATE_FEATURE_ID", "2993113721bc4ffda8728940f29db099"
)
TEMPLATE_VALIDATE_ID = env_template_id(
    "TEMPLATE_VALIDATE_ID", "3233bedf2f7f4e8b82f423ea7cbcdeb7"
)
TEMPLATE_DRIFT_ID = env_template_id(
    "TEMPLATE_DRIFT_ID", "2a0200dcf6f149b68f74710e0e4df2b9"
)
TEMPLATE_HPO_ID = env_template_id("TEMPLATE_HPO_ID", "2bea0fdf45574adc85defd2c7664db39")
TEMPLATE_TRAIN_ID = env_template_id(
    "TEMPLATE_TRAIN_ID", "78ecc3cdb3e243c9a50d2540462c8760"
)
TEMPLATE_EVALUATE_ID = env_template_id(
    "TEMPLATE_EVALUATE_ID", "b5eb901773844e90a81c95159c3addac"
)
TEMPLATE_REGISTER_ID = env_template_id(
    "TEMPLATE_REGISTER_ID", "4f06f43fe9324b18bc77c0e7a6bfa07d"
)
TEMPLATE_COMPARE_CHAMPION_ID = env_template_id(
    "TEMPLATE_COMPARE_CHAMPION_ID", "e577681921ed46748818cde8b6ed0d39"
)
TEMPLATE_PROMOTE_CHAMPION_ID = env_template_id(
    "TEMPLATE_PROMOTE_CHAMPION_ID", "d6dbbd6db51543519b1f7419e37f41e0"
)
TEMPLATE_INFERENCE_ID = env_template_id(
    "TEMPLATE_INFERENCE_ID", "760f866000024b6ea113f16a25ff7efc"
)
TEMPLATE_MONITORING_ID = env_template_id(
    "TEMPLATE_MONITORING_ID", "d140d1257d864c1b81153ab7c0e4a9c6"
)
TEMPLATE_ALERTING_ID = env_template_id(
    "TEMPLATE_ALERTING_ID", "01ce8ba24e53459699c62ef0bf73d9cd"
)
TEMPLATE_AUTO_RETRAINING_ID = env_template_id(
    "TEMPLATE_AUTO_RETRAINING_ID", "8cc49a3858fd410ca1fb8a716cf9e13e"
)
TEMPLATE_EXPLAIN_ID = env_template_id(
    "TEMPLATE_EXPLAIN_ID", "ad2122287ada4243bb59329119877afb"
)
TEMPLATE_DEPLOY_SERVING_ID = env_template_id(
    "TEMPLATE_DEPLOY_SERVING_ID", "ea4f007faec04bce9978096e022efcba"
)
TEMPLATE_VERIFY_ENDPOINT_ID = env_template_id(
    "TEMPLATE_VERIFY_ENDPOINT_ID", "11d3339e65384b3bae63fa1cd263d4ef"
)
TEMPLATE_DEPLOY_CANDIDATE_SERVING_ID = env_template_id(
    "TEMPLATE_DEPLOY_CANDIDATE_SERVING_ID", "69b95d25dabe42dfa25d26873301a1ed"
)
TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_ID = env_template_id(
    "TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_ID", "63af3a1c6fe945c483d7839ee947d2fb"
)

TRAINING_PIPELINE_ID = os.getenv(
    "TRAINING_PIPELINE_ID", "e5f83834496a445ab525e4f46219faf9"
)
# =====================================================
# Utility Functions
# =====================================================


def template_id_map():
    """Return all template IDs configured for the project."""

    return {
        "TEMPLATE_EXTRACT_ID": TEMPLATE_EXTRACT_ID,
        "TEMPLATE_FEATURE_ID": TEMPLATE_FEATURE_ID,
        "TEMPLATE_VALIDATE_ID": TEMPLATE_VALIDATE_ID,
        "TEMPLATE_DRIFT_ID": TEMPLATE_DRIFT_ID,
        "TEMPLATE_HPO_ID": TEMPLATE_HPO_ID,
        "TEMPLATE_TRAIN_ID": TEMPLATE_TRAIN_ID,
        "TEMPLATE_EVALUATE_ID": TEMPLATE_EVALUATE_ID,
        "TEMPLATE_REGISTER_ID": TEMPLATE_REGISTER_ID,
        "TEMPLATE_COMPARE_CHAMPION_ID": TEMPLATE_COMPARE_CHAMPION_ID,
        "TEMPLATE_PROMOTE_CHAMPION_ID": TEMPLATE_PROMOTE_CHAMPION_ID,
        "TEMPLATE_INFERENCE_ID": TEMPLATE_INFERENCE_ID,
        "TEMPLATE_MONITORING_ID": TEMPLATE_MONITORING_ID,
        "TEMPLATE_ALERTING_ID": TEMPLATE_ALERTING_ID,
        "TEMPLATE_AUTO_RETRAINING_ID": TEMPLATE_AUTO_RETRAINING_ID,
        "TEMPLATE_EXPLAIN_ID": TEMPLATE_EXPLAIN_ID,
        "TEMPLATE_DEPLOY_SERVING_ID": TEMPLATE_DEPLOY_SERVING_ID,
        "TEMPLATE_VERIFY_ENDPOINT_ID": TEMPLATE_VERIFY_ENDPOINT_ID,
        "TEMPLATE_DEPLOY_CANDIDATE_SERVING_ID": TEMPLATE_DEPLOY_CANDIDATE_SERVING_ID,
        "TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_ID": TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_ID,
    }


def validate_config(required_ids=None):
    """
    Validate that all required template IDs are populated.
    Call this at the start of pipeline scripts.
    """
    required_ids = required_ids or template_id_map()

    missing = [k for k, v in required_ids.items() if not v]
    if missing:
        raise ValueError(
            f"❌ Missing template IDs: {missing}\n"
            f"   Run: python register_templates.py\n"
            f"   Then copy the printed task IDs into config.py"
        )

    print("✅ All template IDs populated correctly")


# =====================================================
# Logging & Debugging
# =====================================================

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

if DEBUG:
    print("🔧 DEBUG MODE ON")
    print(f"   RANDOM_STATE={RANDOM_STATE}")
    print(f"   N_TRIALS={N_TRIALS}")
    print(f"   MAPE_THRESHOLD={MAPE_THRESHOLD}")
