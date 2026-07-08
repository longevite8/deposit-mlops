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

DEPLOYMENT_VERSION = "v1"
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

CPU_QUEUE = "cpu_queue"
SERVICES_QUEUE = "services"

# =====================================================
# Data Validation Thresholds
# =====================================================

MIN_VALUE = 0.0  # Giá trị tối thiểu cho TARGET_COLUMN (cashflow không âm)
MAX_VALUE = 100000.0  # Giá trị tối đa (outlier detection)

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

DRIFT_PVALUE_THRESHOLD = 0.05  # KS test p-value threshold
DRIFT_RATIO_THRESHOLD = 0.3  # If > 30% of features drift → FAIL
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
ALERT_SMTP_PASSWORD = os.environ.get(
    "CLEARML_ALERT_SMTP_PASSWORD", ""
)  # Never hardcode — use env var!
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

# =====================================================
# Template Task IDs (Populated by register_templates.py)
# =====================================================

TEMPLATE_EXTRACT_ID = "e427c11fc7d54a74b0c8ce2f35beb15b"
TEMPLATE_FEATURE_ID = "22e19644c76b4d16aff3c9deba57f1ad"
TEMPLATE_VALIDATE_ID = "b5c7ec37ef404c1f8e0a6919ec7ccb7e"
TEMPLATE_DRIFT_ID = "5ecaeda02b384600a17dfa50616ce1e0"
TEMPLATE_HPO_ID = "5b9bc840441445a5ac01916c43a3eebf"
TEMPLATE_TRAIN_ID = "725c26ee3cf644368d2e03585a7c31dd"
TEMPLATE_EVALUATE_ID = "f4a1f742c41748acb6414f678d76e2a6"
TEMPLATE_REGISTER_ID = "63a8eddbc7674c35bb17974f62ee6488"
TEMPLATE_COMPARE_CHAMPION_ID = "849019012966454e9b1c370e5ed4e1a4"
TEMPLATE_PROMOTE_CHAMPION_ID = "9cfdc23bcb1841169443bdf4f56be95e"
TEMPLATE_INFERENCE_ID = "0dd2112497f2446e9ca22b5fca1a12a6"
TEMPLATE_MONITORING_ID = "75a1964eb7b8497a8944d3626c47f81c"
TEMPLATE_ALERTING_ID = "8a99db88387e4057bf129ce38dd8c633"
TEMPLATE_AUTO_RETRAINING_ID = "8ceff5c10ec145e4b8d7d5ce395e16f3"
TEMPLATE_EXPLAIN_ID = "2cd2d99b1b044491ab23951a3dbc4b9c"

TRAINING_PIPELINE_ID = "a778c3d31ecd41009f25d8d6184f61d2"
# =====================================================
# Utility Functions
# =====================================================


def validate_config():
    """
    Validate that all required template IDs are populated.
    Call this at the start of pipeline scripts.
    """
    required_ids = {
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
    }

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
