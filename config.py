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

TEMPLATE_EXTRACT_ID = "cdcbb13ee1da4430b6ca2530250835e5"
TEMPLATE_FEATURE_ID = "82c2f47ca09e473d9845ea6cfda20d21"
TEMPLATE_VALIDATE_ID = "0f2cd02de59e4d229695fd537618ed62"
TEMPLATE_DRIFT_ID = "e147e5aac81a43c5a7d80b8715eefad5"
TEMPLATE_HPO_ID = "dd37c7ac706446f687ee0812b0a5f345"
TEMPLATE_TRAIN_ID = "e93feea241ea454ebdab4da94884fdf7"
TEMPLATE_EVALUATE_ID = "0e597089b2994e868f133efe5c7f9f02"
TEMPLATE_REGISTER_ID = "d48934f3f1e344fcb5e2cca935615ef0"
TEMPLATE_COMPARE_CHAMPION_ID = "69f9e5d71adc432a8335b372ff6cbf1a"
TEMPLATE_PROMOTE_CHAMPION_ID = "2a41222842fa4116ba01ff7ba93d0f1e"
TEMPLATE_INFERENCE_ID = "82edbdbaf6224d3aaa03e0ee21623c59"
TEMPLATE_MONITORING_ID = "5618bffaa02d41928a16fe13a574156a"
TEMPLATE_ALERTING_ID = "67c47800d8da4f64bed72e71c61025ed"
TEMPLATE_AUTO_RETRAINING_ID = "7af8dd8affe34bb8b565454d85e3852c"
TEMPLATE_EXPLAIN_ID = "95b9540054e64981a6471a9c5328305b"

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

TEMPLATE_HPO_LIGHTGBM_ID = "de71069fa07947a4b9ec9cfdd6702a75"
TEMPLATE_HPO_NBEATSX_ID = "1c0266e7b39347e8bbefc0c61359ded1"
TEMPLATE_HPO_NHITS_ID = "10bba8bb4eb34454a8bfd40061936973"
TEMPLATE_COMPARE_HPO_ID = "2de6650563494a52a3b514914ca4e574"

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
