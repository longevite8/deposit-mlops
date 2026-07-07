# Cashflow Forecasting MLOps with ClearML

End-to-end MLOps project for cashflow forecasting using ClearML with **Champion/Challenger architecture**, **dual pipelines** (Training + Production), and **automated retraining**.

The project covers:

- **Feature Engineering** - Lag and rolling window features
- **Data Validation** - Schema, dtype, range, and missing value checks
- **Drift Detection** - Kolmogorov-Smirnov statistical test
- **Hyperparameter Optimization** - Optuna-based HPO with best parameters tracking
- **Model Training** - LightGBM with training loss curves and feature importance
- **Model Evaluation** - Multi-metric evaluation with quality gates
- **Model Explainability** - SHAP TreeExplainer for feature importance analysis
- **Model Registry** - ClearML model registry with tagging (candidate/champion/archived)
- **Production Inference** - Batch predictions with performance monitoring
- **Monitoring** - Real-time MAPE/R² tracking and drift monitoring
- **Email Alerting** - SMTP-based alerts with configurable thresholds
- **Continuous Retraining** - Fire-and-forget auto-retraining with Champion Registry
- **Champion/Challenger Architecture** - Compare candidate vs champion metrics

---

## Project Structure

```text
cashflow-clearml/
│
├── pipelines/
│   ├── trainning_pipeline.py          # Training DAG (10 steps with caching)
│   └── production_pipeline.py          # Production DAG (7 steps, live state)
│
├── tasks/                              # 15 ClearML task templates
│   ├── extract_data.py                 # Synthetic data generation
│   ├── feature_engineering.py          # Lag & rolling features
│   ├── validate_data.py                # Schema/dtype/range validation
│   ├── drift_detection.py              # KS statistical test
│   ├── hpo_model.py                    # Optuna hyperparameter optimization
│   ├── train_model.py                  # LightGBM training with loss curves
│   ├── evaluate_model.py               # Test set evaluation with quality gate
│   ├── register_model.py               # Model tagging (candidate/champion/archived)
│   ├── compare_champion.py             # Candidate vs champion comparison
│   ├── promote_champion.py             # Champion promotion via APIClient
│   ├── explain_model.py                # SHAP TreeExplainer analysis
│   ├── inference_model.py              # Batch prediction with latency metrics
│   ├── monitoring_model.py             # MAPE/R² tracking & drift monitoring
│   ├── alerting_model.py               # Email alert generation
│   └── auto_retraining.py              # Fire-and-forget pipeline trigger
│
├── src/
│   └── features.py                     # Feature engineering functions
│
├── .github/
│   └── workflows/
│       └── test.yml                    # GitHub Actions CI/CD
│
├── docs/                               # Documentation
│   ├── architecture.md                 # System architecture
│   ├── training_pipeline.md            # Training DAG details
│   ├── production_pipeline.md          # Production DAG details
│   ├── monitoring_and_drift.md         # Monitoring & drift detection
│   ├── champion_registry.md            # Champion registry pattern
│   ├── explainability.md               # SHAP analysis details
│   ├── ci_cd.md                        # GitHub Actions workflow
│   ├── agent_setup.md                  # ClearML Agent configuration
│   └── cleanup.md                      # Resource cleanup guide
│
├── config.py                           # Centralized configuration
├── register_templates.py               # One-time template registration
├── delete_tasks.py                     # ClearML resource cleanup utility
├── .env                                # Environment variables (not committed)
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project metadata
└── README.md                           # This file
```

---

## Documentation

### Getting Started

- **[architecture.md](docs/architecture.md)** - System architecture with Champion/Challenger design
- **[training_pipeline.md](docs/training_pipeline.md)** - 10-step training DAG with caching strategy
- **[production_pipeline.md](docs/production_pipeline.md)** - 7-step production DAG with auto-retraining

### Core Features

- **[explainability.md](docs/explainability.md)** - SHAP TreeExplainer for model interpretability
- **[monitoring_and_drift.md](docs/monitoring_and_drift.md)** - Real-time monitoring and drift detection
- **[champion_registry.md](docs/champion_registry.md)** - Singleton model registry pattern

### Operations

- **[ci_cd.md](docs/ci_cd.md)** - GitHub Actions CI/CD workflow
- **[agent_setup.md](docs/agent_setup.md)** - ClearML Agent configuration for distributed execution
- **[cleanup.md](docs/cleanup.md)** - Resource cleanup and project management

---

## Training Pipeline

The training pipeline produces candidate models through a multi-step workflow:

```text
Extract → Feature → Validate → Drift → HPO → Train → Evaluate → Register → Explain → Compare → Promote
```

**Key Steps:**
- **Extract**: Synthetic cashflow data generation with Gamma distribution
- **Feature**: Create lag (1, 7) and rolling mean (7) features
- **Validate**: Data quality checks (schema, dtypes, ranges, missing values)
- **Drift**: KS test vs champion reference dataset (graceful on first run)
- **HPO**: Optuna hyperparameter optimization with best params tracking
- **Train**: LightGBM training with loss curves and feature importance metrics
- **Evaluate**: Test set evaluation with MAPE/R² quality gates
- **Register**: Model tagging as "candidate" in ClearML registry
- **Explain**: SHAP TreeExplainer analysis with top-3 feature scalars
- **Compare**: Candidate vs champion metrics comparison
- **Promote**: Champion tag update via APIClient (if candidate wins)

**Caching Strategy** (Deterministic Steps):
- Extract, Feature, Validate, Drift, HPO: Cached (parameters unchanged)
- Train, Evaluate, Register, Compare, Promote: No cache (model state dependent)

---

## Production Pipeline

The production pipeline runs inference and monitoring on fresh data:

```text
Extract → Feature → Drift & Inference → Monitoring → Alerting → Auto-Retraining
```

**Key Steps:**
- **Extract**: Load latest production data
- **Feature**: Apply same feature engineering as training
- **Drift**: KS test vs champion training reference
- **Inference**: Batch predictions with champion model (latency metrics)
- **Monitoring**: MAPE/R² tracking + drift ratio analysis
- **Alerting**: Generate email alerts if thresholds exceeded
- **Auto-Retraining**: Fire-and-forget trigger for training pipeline (if alert generated)

**Caching Strategy** (Live State):
- Extract, Feature: Cached (deterministic)
- Drift, Inference, Monitoring, Alerting, Auto-Retraining: No cache (live monitoring)

---

## Main Technologies

- **Python 3.10+** - Core language
- **ClearML 1.4.0+** - ML orchestration and experiment tracking
  - WebApp 2.4.0+ for UI
  - Agent 1.4.0+ for distributed execution
- **LightGBM** - Gradient boosting model
- **Optuna** - Hyperparameter optimization
- **SHAP** - Model explainability and feature importance
- **Pandas/NumPy** - Data processing
- **scikit-learn** - Data validation utilities
- **GitHub Actions** - CI/CD automation
- **SMTP** - Email alerting via Gmail App Password

---

## Key Features

### 1. Champion/Challenger Architecture

Two-pipeline design ensures production stability:
- **Training Pipeline**: Produces candidate models, evaluates against champion
- **Production Pipeline**: Always uses current champion for inference
- **Champion Registry**: Singleton ClearML task tracking the champion model metadata

### 2. Two-Artifact Lineage Pattern

Every task uploads two artifacts:
- `<task>_summary`: Decision/result (e.g., `{"candidate_win": True}`)
- `<task>_lineage`: Upstream task IDs for full traceability

### 3. ClearML Dataset-Based Data Exchange

Features flow through **ClearML Dataset objects** (not direct DataFrame artifacts):
- Feature task creates parquet files → ClearML Dataset
- Downstream tasks load via `Dataset.get_local_copy()`
- Ensures reproducibility and version tracking

### 4. Email Alerting with SMTP

Configurable threshold-based alerts via Gmail App Password:
- MAPE exceeds threshold
- R² drops below threshold  
- Data drift detected (KS test fails)
- Triggered in production pipeline → auto-retraining

### 5. Model Explainability (SHAP)

Post-training explainability analysis:
- TreeExplainer for LightGBM models
- Feature importance rankings
- SHAP dependence plots
- Top-3 feature scalars in ClearML dashboard

### 6. Comprehensive Monitoring

Real-time production monitoring:
- Prediction MAPE and R² tracking
- Data drift detection via KS test
- Batch inference latency metrics
- Dashboard-ready markdown summaries

### 7. GitHub Actions CI/CD

Automated validation pipeline:
- Linting (flake8) for code quality
- Unit tests (pytest) with coverage
- Codecov integration
- ClearML template validation

---

## Quick Start

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Configure ClearML

Create `.env` file with:

```bash
CLEARML_WEB_HOST=http://localhost:8080
CLEARML_API_HOST=http://localhost:8008
CLEARML_FILES_HOST=http://localhost:8081
CLEARML_API_ACCESS_KEY=your-access-key
CLEARML_API_SECRET_KEY=your-secret-key
ALERT_SMTP_PASSWORD=your-gmail-app-password
GIT_REPO=https://github.com/longevite8/cashflow-clearml.git
```

### 3. Register Task Templates

```bash
python register_templates.py
```

Copy printed template IDs into `config.py` as `TEMPLATE_<TASK>_ID`.

### 4. Run Training Pipeline

```bash
python pipelines/trainning_pipeline.py
```

### 5. Run Production Pipeline

```bash
python pipelines/production_pipeline.py
```

### 6. Cleanup Resources (Optional)

```bash
python delete_tasks.py
```

---

## Configuration

All configuration centralized in `config.py`:

- **Project Names**: `PROJECT_PARENT = "Deposit-CashFlow"` (hierarchical)
- **Queue Names**: `CPU_QUEUE = "cpu_queue"`, `SERVICES_QUEUE = "services"`
- **Template IDs**: All 15 task templates (populated after `register_templates.py`)
- **Feature Columns**: `LAG_FEATURES = [1, 7]`, `ROLLING_FEATURES = [("rolling_mean", 7)]`
- **Validation Thresholds**: Missing rate, min/max values
- **Monitoring Thresholds**: MAPE, R², drift p-value, drift ratio
- **Email Alerting**: SMTP host, port, sender, recipient
- **SHAP Parameters**: Sample size, max display features

---

## ClearML Agent Setup

Run distributed pipeline tasks on managed agents:

```bash
# Configure agent
clearml-agent daemon --queue cpu_queue

# Or with Docker
clearml-agent daemon --queue cpu_queue --docker nvidia/cuda:11.8.0-runtime-ubuntu22.04
```

See [agent_setup.md](docs/agent_setup.md) for detailed configuration.

---

## Troubleshooting

### Training Pipeline Hangs

Check if auto-retraining is running on same queue:
- Auto-retraining uses `SERVICES_QUEUE` (fire-and-forget pattern)
- Training uses `CPU_QUEUE`
- Ensure both agents are active

### Email Alerts Not Sending

1. Verify SMTP credentials in `.env`
2. Check `ALERT_SMTP_PASSWORD` is Gmail **App Password** (not regular password)
3. Verify recipient email in `config.py` as `ALERT_EMAIL_TO`

### Drift Detection Fails on First Run

Graceful handling implemented - first run uses current data as reference, always `PASS`.

### Model Tags Not Updated

Use `APIClient` for tag mutations (not property setter):
```python
from clearml.backend_api.session.client import APIClient
client = APIClient()
client.models.edit(model=model_id, tags=new_tags)
```

---

## Documentation References

For detailed information on each component, see the [docs/](docs/) folder.

---

## License

MIT License
