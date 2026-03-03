# 21f3001527_MLOPS_WEEKLY_ASSIGNMENT

# Week 4 - Integrating CI into the IRIS ML Pipeline

## Overview
This week we integrated Continuous Integration (CI) into the IRIS classification pipeline using GitHub Actions. The CI pipeline automatically runs unit tests, trains the model, and publishes test reports as comments on every push and pull request using CML (Continuous Machine Learning).

---

## Repository Structure
```
21f3001527_MLOPS_WEEKLY_ASSIGNMENT/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI workflow
├── .dvc/                   # DVC configuration
├── artifacts.dvc           # DVC tracked model artifacts
├── data.dvc                # DVC tracked dataset
├── model.pkl               # Trained Random Forest model
├── test_model.py           # Unit tests for sanity testing
├── train.py                # Model training script
├── requirements.txt        # Python dependencies
└── README.md
```

---

### Part A - Codebase in GitHub Repo
- Verified Week 2 codebase was available in `week_02` branch
- Created `week_04` branch from `week_02`
- Pushed all code to GitHub remote repository

### Part B - Unit Tests for Sanity Testing
- Trained the IRIS classification model using `train.py`
- Model saved as `model.pkl` using `joblib`
- Wrote unit tests in `test_model.py` using `unittest` framework
- 3 tests written and verified locally:
  - `test_model_loads` - checks model loads correctly
  - `test_single_prediction` - checks prediction on single sample
  - `test_batch_prediction` - checks predictions on multiple samples
- All 3 tests passed locally before pushing to GitHub

### Part C - GitHub Actions CI Workflow
- Created `.github/workflows/ci.yml`
- Workflow triggers on:
  - Every push to any branch
  - Every pull request to any branch
- CI pipeline steps:
  1. Checkout code
  2. Set up Python 3.9
  3. Install dependencies (pytest, scikit-learn, numpy, joblib, pandas)
  4. Train the model
  5. Run sanity tests
  6. Setup Node.js for CML
  7. Post CML test report as comment on commit/PR

---

## CI Workflow

```yaml
name: IRIS ML CI

on:
  push:
    branches:
      - '**'
  pull_request:
    branches:
      - '**'

permissions:
  contents: write
  pull-requests: write

jobs:
  test-and-report:
    runs-on: ubuntu-latest
    steps:
      - Checkout Code
      - Set Up Python 3.9
      - Install Dependencies
      - Train Model
      - Run Sanity Tests
      - Setup Node
      - Post CML Report as Comment
```

---

## Model Training

The `train.py` script:
- Loads the IRIS dataset from `sklearn`
- Splits data into train/test sets (80/20)
- Trains a `RandomForestClassifier`
- Evaluates accuracy on test set
- Saves model as `model.pkl` using `joblib`
- Saves metrics to `metrics.txt`

**Achieved Accuracy: ~96.67%**

---

## Unit Tests

The `test_model.py` file contains:

```python
class TestIrisModel(unittest.TestCase):

    def test_model_loads(self):
        # Checks model loads without errors

    def test_single_prediction(self):
        # Checks single sample prediction returns valid class [0, 1, 2]

    def test_batch_prediction(self):
        # Checks batch of 3 samples returns 3 predictions
```

### Running Tests Locally
```bash
python -m pytest test_model.py -v
```

### Expected Output
```
test_model.py::TestIrisModel::test_batch_prediction PASSED   [ 33%]
test_model.py::TestIrisModel::test_model_loads PASSED        [ 66%]
test_model.py::TestIrisModel::test_single_prediction PASSED  [100%]
================ 3 passed in 1.38s ================
```

---

## CML Report
CML (Continuous Machine Learning) automatically posts test results as a comment on every commit and pull request.

Example CML comment posted automatically:
```
## Sanity Test Report
===== test session starts =====
test_model.py::TestIrisModel::test_batch_prediction PASSED   [ 33%]
test_model.py::TestIrisModel::test_model_loads PASSED        [ 66%]
test_model.py::TestIrisModel::test_single_prediction PASSED  [100%]
================ 3 passed in 1.14s ================
```

---

## Pull Request Flow
1. All week 4 work done on `week_04` branch
2. Pull Request created: `week_04` → `main`
3. CI automatically triggered on PR
4. Both checks passed:
   - IRIS ML CI / test-and-report (push) ✅
   - IRIS ML CI / test-and-report (pull_request) ✅
5. CML report posted as comment on PR
6. PR successfully merged into `main`

---

## How CI Fits Into the Pipeline

```
Git Push / PR
    ↓
GitHub Actions
    ↓
Train Model → Run pytest → CML Report
    ↓
Validated Pipeline ✅
```

---

## Tech Stack
- **Python** 3.9
- **scikit-learn** - Model training
- **joblib** - Model serialization
- **pytest / unittest** - Unit testing
- **GitHub Actions** - CI/CD
- **CML** - ML test reporting
- **DVC** - Data and model versioning
- **GCS** - Remote storage

---