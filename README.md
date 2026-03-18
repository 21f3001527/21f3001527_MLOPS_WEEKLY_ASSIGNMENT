# 21f3001527_MLOPS_WEEKLY_ASSIGNMENT

# Week 5 - Integrating MLflow into the IRIS ML Pipeline

## Overview
This week we integrated MLflow experiment tracking and model registry into the IRIS classification pipeline. The pipeline now logs hyperparameters, evaluation metrics, and trained models to MLflow for every training run, enabling experiment comparison and centralized model management.

---

## Repository Structure
```
21f3001527_MLOPS_WEEKLY_ASSIGNMENT/
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI workflow with MLflow
├── .dvc/                   # DVC configuration
├── data.dvc                # DVC tracked dataset (data only)
├── data.csv                # IRIS dataset
├── test_model.py           # Unit tests fetching model from MLflow registry
├── train.py                # Model training script with MLflow tracking
├── requirements.txt        # Python dependencies
└── README.md
```

---

## What Changed from Week 4

| Component | Week 4 | Week 5 |
|-----------|--------|--------|
| Model storage | DVC (`artifacts.dvc`) | MLflow Model Registry |
| Experiment tracking | None | MLflow |
| Hyperparameter tuning | Single run | 6 combinations |
| Model loading in tests | Local `model.pkl` | MLflow Registry |
| CI model source | DVC | MLflow (sqlite) |

---

### Task 1 - Hyperparameter Tuning
- Varied `n_estimators`: [50, 100, 200]
- Varied `max_depth`: [3, 5]
- Produced **6 training runs** with different configurations

### Task 2 - Log Experiments with MLflow
For each run logged:
- **Parameters**: n_estimators, max_depth, random_state
- **Metrics**: accuracy, precision, f1_score
- **Model**: saved as MLflow artifact
- **Tag**: Training Info

### Task 3 - Compare Experiments in MLflow UI
- Ran 6 experiments with different hyperparameter configurations
- Compared metrics side-by-side in MLflow Tracking UI
- Best accuracy achieved: **100%** (n_estimators=200, max_depth=5)

### Task 4 - Remove Model Dependency from DVC
- Removed `artifacts.dvc` and `model.pkl` from DVC tracking
- Models now stored exclusively in MLflow Model Registry
- DVC continues to track `data.csv` only

### Task 5 - Fetch Models from MLflow for Evaluation
- Updated `test_model.py` to load model from MLflow registry
- Model resolved by registered name: `IRIS-classifier-rf`
- No local path or DVC dependency

### Task 6  - MLflow in CI
- Updated GitHub Actions CI to use MLflow with sqlite tracking
- CI trains model → logs to MLflow → runs tests against registry model
- CML posts test report as comment on every push/PR

---

## MLflow Experiment

### Tracking URI
```
http://localhost:8100  # local
sqlite:///mlflow.db    # CI
```

### Experiment Name
```
MLflow with IRIS Dataset
```

### Registered Model
```
IRIS-classifier-rf
```

---

## Model Training

The `train.py` script:
- Loads the IRIS dataset from `sklearn`
- Splits data into train/test sets (80/20)
- Runs hyperparameter tuning across 6 combinations
- Logs each run to MLflow with params, metrics and model
- Registers best model in MLflow Model Registry

---

## Unit Tests
```python
class TestIrisModel(unittest.TestCase):

    def test_model_loads(self):
        # Loads model from MLflow registry

    def test_single_prediction(self):
        # Checks prediction on single sample

    def test_batch_prediction(self):
        # Checks predictions on multiple samples

    def test_prediction_accuracy(self):
        # Checks accuracy > 90%
```

### Running Tests Locally
```bash
python -m pytest test_model.py -v
```

### Expected Output
```
test_model.py::TestIrisModel::test_batch_prediction PASSED      [ 25%]
test_model.py::TestIrisModel::test_model_loads PASSED           [ 50%]
test_model.py::TestIrisModel::test_prediction_accuracy PASSED   [ 75%]
test_model.py::TestIrisModel::test_single_prediction PASSED     [100%]
================ 4 passed in 3.16s ================
```

---

## CI Workflow
```
Git Push / PR
    ↓
GitHub Actions
    ↓
Install Dependencies
    ↓
Train Model → Log to MLflow (sqlite)
    ↓
Fetch Model from MLflow Registry
    ↓
Run pytest → CML Report
    ↓
Validated Pipeline ✅
```

---

## Tech Stack
- **Python** 3.10
- **scikit-learn** - Model training
- **MLflow** - Experiment tracking and model registry
- **pytest / unittest** - Unit testing
- **GitHub Actions** - CI/CD
- **CML** - ML test reporting
- **DVC** - Data versioning
- **GCS** - Remote storage

---
