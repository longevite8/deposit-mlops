# GitHub Actions CI/CD Workflow

## Overview

The project includes an automated **GitHub Actions workflow** (`test.yml`) that validates code quality, runs unit tests, and verifies ClearML template integrity on every push.

---

## Workflow File

Location: `.github/workflows/test.yml`

---

## Pipeline Steps

### 1. Code Linting (flake8)

Enforces code quality standards:

```bash
flake8 . --max-line-length=100 --exclude=.venv,__pycache__
```

**Checks:**

* PEP 8 compliance
* Unused imports
* Variable naming conventions
* Code complexity

**Configuration:**

* Max line length: 100 characters
* Excludes: `.venv/`, `__pycache__/`

---

### 2. Unit Tests (pytest)

Runs test suite with coverage reporting:

```bash
pytest tests/ --cov=. --cov-report=xml --cov-report=term
```

**Features:**

* Discovers tests in `tests/` directory
* Generates XML coverage report
* Displays terminal coverage summary
* Fails if any test fails

**Expected Coverage:**

* Aim for >80% coverage on new code
* Critical paths should have >95% coverage

---

### 3. Coverage Upload (Codecov)

Uploads coverage reports to Codecov:

```
codecov-action
  ↓
Upload XML coverage to codecov.io
  ↓
Merge with historical data
  ↓
Display coverage badge in README
```

**Benefits:**

* Track coverage trends over time
* PR comments on coverage changes
* Badges for README and badges

---

### 4. ClearML Template Validation

Validates all task templates register correctly:

```bash
python register_templates.py --validate
```

**Checks:**

* All task files found
* Task signatures correct
* Template parameters valid
* ClearML project connectivity

---

## Trigger Events

Workflow runs on:

* **Push to main branch**
* **Pull requests to main branch**
* **Manual trigger** (GitHub Actions tab)

---

## GitHub Secrets Configuration

The workflow uses GitHub Secrets for sensitive credentials:

### Required Secrets

1. **CLEARML_API_ACCESS_KEY**
   * Source: ClearML Web UI → Settings → API Credentials
   * Value: Your ClearML API access key

2. **CLEARML_API_SECRET_KEY**
   * Source: ClearML Web UI → Settings → API Credentials
   * Value: Your ClearML API secret key

### Setup Instructions

1. Go to GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add each secret:
   * Name: `CLEARML_API_ACCESS_KEY`
   * Value: (paste from ClearML)
5. Repeat for `CLEARML_API_SECRET_KEY`

---

## Workflow Output

### Success Case

```
test.yml: Linting passed
  │ flake8 found 0 issues
  │
test.yml: Tests passed
  │ pytest 42 passed in 3.21s
  │ Coverage: 86%
  │
test.yml: Coverage uploaded
  │ Codecov received coverage report
  │
test.yml: Templates validated
  │ All 15 templates registered successfully
```

### Failure Cases

**Linting failure:**

```
test.yml: Linting failed
  │ flake8 found 5 issues:
  │   - config.py:10: E501 line too long
  │   - tasks/train_model.py:25: F841 unused variable
```

**Test failure:**

```
test.yml: Tests failed
  │ pytest 40 passed, 2 failed
  │   FAILED test_feature_engineering.py::test_lag_creation
  │   FAILED test_drift_detection.py::test_ks_test
```

**Template validation failure:**

```
test.yml: Template validation failed
  │ Error: TEMPLATE_EXPLAIN_ID not found in config.py
  │ Action: Run register_templates.py and copy IDs to config.py
```

---

## Local Workflow Simulation

### Run Linting Locally

```bash
pip install flake8
flake8 . --max-line-length=100 --exclude=.venv,__pycache__
```

### Run Tests Locally

```bash
pip install pytest pytest-cov
pytest tests/ --cov=. --cov-report=term
```

### Validate Templates Locally

```bash
python register_templates.py --validate
```

---

## Common Issues & Solutions

### Issue 1: "Secrets not found"

**Symptom:** Workflow fails with `CLEARML_API_ACCESS_KEY not set`

**Solution:**

1. Verify GitHub Secrets are created
2. Check secret names match exactly (case-sensitive)
3. Wait 60 seconds for secret propagation
4. Re-run workflow

### Issue 2: "Linting errors on unchanged code"

**Symptom:** flake8 fails for existing code

**Solution:**

```bash
# Fix automatically
black . --line-length=100

# Or run flake8 with fixes
autopep8 -i tasks/*.py src/*.py
```

### Issue 3: "Test timeout"

**Symptom:** `pytest` hangs or exceeds 10-minute limit

**Solution:**

```bash
# Add timeout decorator to test
import pytest

@pytest.mark.timeout(5)  # 5-second timeout
def test_slow_operation():
    ...
```

### Issue 4: "Coverage threshold not met"

**Symptom:** Coverage drops below 80%

**Solution:**

* Add tests for new code
* Use `pytest --cov-report=html` to see gaps
* Increase test coverage in PR

---

## Extending the Workflow

### Add New Linting Rules

Edit `.flake8` or add to `pyproject.toml`:

```toml
[tool.flake8]
max-line-length = 100
extend-ignore = ["E203", "W503"]
exclude = [".venv", "__pycache__"]
```

### Add Custom Test Step

```yaml
- name: Run integration tests
  run: |
    pytest tests/integration/ -v --timeout=60
```

### Add Security Scanning

```yaml
- name: Run Bandit security scan
  run: |
    pip install bandit
    bandit -r tasks/ src/ --severity-level medium
```

### Add Type Checking

```yaml
- name: Type check with mypy
  run: |
    pip install mypy
    mypy tasks/ src/ --ignore-missing-imports
```

---

## Best Practices

### 1. Keep Tests Fast

* Aim for <5 seconds per test
* Use mocks for external calls
* Run locally before pushing

### 2. Maintain Coverage

* Add tests for new features
* Target >85% overall coverage
* Comment critical untestable code

### 3. Fail Fast

* Linting runs first (fastest)
* Tests run next (medium)
* Template validation last (slowest)

### 4. Document Changes

* Update README when adding steps
* Comment complex test scenarios
* Explain workflow triggers

---

## Workflow Timeline

Typical workflow execution:

```
Start (push to main)
    ↓
Checkout code (30s)
    ↓
Setup Python 3.11 (20s)
    ↓
Install dependencies (45s)
    ↓
Run flake8 (30s)
    ↓
Run pytest (120s)
    ↓
Upload coverage (15s)
    ↓
Validate templates (60s)
    ↓
Complete (320s total ≈ 5 minutes)
```

---

## GitHub Actions Dashboard

View workflow results:

1. Go to repository
2. Click "Actions" tab
3. Select latest workflow run
4. View step-by-step logs
5. Check artifact outputs (coverage HTML, etc.)

---

## Integration with PR Reviews

### Coverage Comment on PR

Codecov automatically comments on PRs:

```
Codecov Report

Coverage: 86% (↑2% from main)
Changes:
  + tasks/explain_model.py: 92% coverage
  + tests/test_explain.py: 95% coverage

✓ All checks passed
```

### Required Status Checks

Configure in GitHub:

1. Settings → Branches → Add rule
2. Require status check:
   * `test / lint`
   * `test / pytest`
   * `test / templates`

Prevents merging without passing checks.

---

## Troubleshooting

### Workflow Won't Trigger

Check:

* Is branch `main`? (configured in workflow)
* Is workflow enabled? (Actions tab → Enable)
* Did you wait 60s after enabling?

### Inconsistent Results (Local vs CI)

Possible causes:

* Python version difference
* Dependency version difference
* Environment variables not set

Solution:

```bash
# Match CI Python version
python3.11 -m venv .venv_ci
source .venv_ci/bin/activate
pip install -r requirements.txt
```

---

## References

* [GitHub Actions Documentation](https://docs.github.com/en/actions)
* [pytest Documentation](https://docs.pytest.org/)
* [flake8 Documentation](https://flake8.pycqa.org/)
* [Codecov GitHub Actions](https://github.com/codecov/codecov-action)
