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

PROJECT_PARENT = "Recovery-CashFlow"
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

# =====================================================
# Template Task IDs (Populated by register_templates.py)
# =====================================================

TEMPLATE_EXTRACT_ID = "9c7bec49aab9497f917fb046e15144f4"
TEMPLATE_FEATURE_ID = "653cfaf198d047aea12ec1548ff028cf"
TEMPLATE_VALIDATE_ID = "e66b1c1e7713438cb3cd0bc0b5a91235"
TEMPLATE_DRIFT_ID = "09f897cd19da4d758437bd23e3c152b8"
TEMPLATE_HPO_ID = "297f94dbe2fe4c299b97ad554eb960e4"
TEMPLATE_TRAIN_ID = "d8f2fea28a2c4e4582db63cb207f843e"
TEMPLATE_EVALUATE_ID = "14602498478e4888a4d2e56e2bcba34d"
TEMPLATE_REGISTER_ID = "49b928a800a34ed88d4ca06261039795"
TEMPLATE_COMPARE_CHAMPION_ID = "9a0f439e733c48e08cb339f790cad935"
TEMPLATE_PROMOTE_CHAMPION_ID = "267d6296de0f461b90d0f294970ea78b"
TEMPLATE_INFERENCE_ID = "64ac0efab5d94e2d8ed7ae204933fbfe"
TEMPLATE_MONITORING_ID = "81a2ed7819f94af09ef4d2dd5ce98276"
TEMPLATE_ALERTING_ID = "98f3947d910945e4bc26d63ec47ec118"
TEMPLATE_AUTO_RETRAINING_ID = "504f52abf25d4fefb720320178a166be"
TEMPLATE_EXPLAIN_ID = "a89b80f2c0644f9892e78bdddb710f46"

TRAINING_PIPELINE_ID = "c1d61d3965f942c6a9fc7736eb67c870"


# =====================================================
# Model Categories
# =====================================================

TREE_MODELS = ["lightgbm"]
NEURAL_MODELS = ["nbeatsx", "nhits"]
SUPPORTED_MODELS = TREE_MODELS + NEURAL_MODELS

# =====================================================
# HPO Pruner Configuration (Dynamic based on SUPPORTED_MODELS)
# =====================================================

HPO_USE_HYPERBAND_BY_MODEL = {
    model: model in NEURAL_MODELS  # True for neural models, False for tree-based
    for model in SUPPORTED_MODELS
}

HYPERBAND_R = 9
HYPERBAND_ETA = 3
HYPERBAND_MIN_RESOURCE = 1
HYPERBAND_MAX_RESOURCE = 100

# =====================================================
# LightGBM Hyperparameter Ranges (HPO Search Space)
# =====================================================

LGBM_LEARNING_RATE_MIN = 0.01
LGBM_LEARNING_RATE_MAX = 0.1
LGBM_NUM_LEAVES_MIN = 15
LGBM_NUM_LEAVES_MAX = 100
LGBM_N_ESTIMATORS_MIN = 100
LGBM_N_ESTIMATORS_MAX = 1000
LGBM_MAX_DEPTH_MIN = 5
LGBM_MAX_DEPTH_MAX = 15
LGBM_EARLY_STOPPING_ROUNDS = 50

# =====================================================
# NBEATSx Configuration (Neural Time Series)
# =====================================================

NBEATSX_INPUT_SIZE = 8  # Temporal window/lookback period
NBEATSX_FORECAST_HORIZON = 1  # Steps ahead to predict
NBEATSX_BATCH_SIZE = 32
NBEATSX_EPOCHS = 100
NBEATSX_LEARNING_RATE = 0.001
NBEATSX_EARLY_STOPPING_PATIENCE = 10

# NBEATSx HPO Search Space
NBEATSX_STACK_SIZES_MIN = 4
NBEATSX_STACK_SIZES_MAX = 16
NBEATSX_HIDDEN_SIZE_MIN = 32
NBEATSX_HIDDEN_SIZE_MAX = 256
NBEATSX_NUM_STACKS = 2
NBEATSX_NUM_LAYERS_MIN = 2
NBEATSX_NUM_LAYERS_MAX = 5

# =====================================================
# NHITS Configuration (Neural Hierarchical Time Series)
# =====================================================

NHITS_INPUT_SIZE = 8
NHITS_FORECAST_HORIZON = 1
NHITS_BATCH_SIZE = 32
NHITS_EPOCHS = 100
NHITS_LEARNING_RATE = 0.001
NHITS_EARLY_STOPPING_PATIENCE = 10

# NHITS HPO Search Space
NHITS_NUM_STACKS_MIN = 2
NHITS_NUM_STACKS_MAX = 5
NHITS_NUM_BLOCKS_MIN = 1
NHITS_NUM_BLOCKS_MAX = 3
NHITS_NUM_LAYERS_MIN = 2
NHITS_NUM_LAYERS_MAX = 5
NHITS_LAYER_WIDTH_MIN = 64
NHITS_LAYER_WIDTH_MAX = 256

# =====================================================
# Normalization (cho Neural Models)
# =====================================================

NORMALIZATION_METHOD = "standard"  # "standard", "minmax", "robust"

# =====================================================
# Template Task Names (Model Selection Pipeline)
# =====================================================

TEMPLATE_HPO_LIGHTGBM_NAME = "HPO LightGBM"
TEMPLATE_HPO_NBEATSX_NAME = "HPO NBEATSx"
TEMPLATE_HPO_NHITS_NAME = "HPO NHITS"
TEMPLATE_COMPARE_HPO_NAME = "Compare HPO Results"

# =====================================================
# Template Task IDs (Model Selection) - Populate after register_templates.py
# =====================================================

TEMPLATE_HPO_LIGHTGBM_ID = "3a699dd0f68b49ef8f53de649ee70e1f"
TEMPLATE_HPO_NBEATSX_ID = "13fe52296bb44e689e4c9f08ee11cefa"
TEMPLATE_HPO_NHITS_ID = "6d132c2a9c13445a90e599f7d83f2d4d"
TEMPLATE_COMPARE_HPO_ID = "bf426e94286c409c89d46baa69515993"

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
