# 21f3001527_MLOPS_WEEKLY_ASSIGNMENT

Reproducible IRIS ML pipeline with DVC version control backed by Google Cloud Storage.

## Objectives

- Version datasets and model artifacts using DVC
- Configure GCS as a DVC remote backend
- Train across multiple data iterations
- Switch between data/model versions using Git tags + DVC checkout

## Setup

```bash
git clone -b week_02 https://github.com/21f3001527/21f3001527_MLOPS_WEEKLY_ASSIGNMENT.git
cd 21f3001527_MLOPS_WEEKLY_ASSIGNMENT

python3 -m venv .env && source .env/bin/activate
pip install -r requirements.txt
pip install dvc[gs]

dvc remote add -d gcsremote gs://<your_bucket>/iris_dvc
dvc remote modify gcsremote credentialpath <path_to_service_account.json>
```

## Workflow

### Iteration 1 — Original IRIS Data

```bash
dvc add data/
python train.py
dvc add artifacts/

git add data.dvc artifacts.dvc .gitignore
git commit -m "First iteration done with original IRIS data"
git tag -a "v1.0" -m "model v1.0 with original data"
```

### Iteration 2 — Augmented IRIS Data

```bash
dvc add data/
python train.py
dvc add artifacts/

git add data.dvc artifacts.dvc .gitignore
git commit -m "Second iteration done with augmented IRIS data"
git tag -a "v2.0" -m "model v2.0 with augmented data"
```

### Switch Between Versions

```bash
# Revert to v1.0
git checkout v1.0
dvc checkout

# Return to latest
git checkout week_02
dvc checkout
```

### Push Everything

```bash
git push origin week_02 --tags
dvc push
```

## Repository Structure

```
data/               # IRIS dataset (DVC-tracked)
artifacts/
├── metrics.txt     # Training metrics (DVC-tracked)
└── model.joblib    # Trained model (DVC-tracked)
artifacts.dvc       # DVC pointer for artifacts
data.dvc            # DVC pointer for dataset
train.py            # Training script
requirements.txt    # Python dependencies
.dvc/               # DVC config
```

## Key Concepts

- **DVC** tracks large files (data, models) that Git cannot handle efficiently
- **Git tags** (`v1.0`, `v2.0`) mark reproducible experiment snapshots
- **`dvc checkout`** restores local files to match the tagged version
- **GCS remote** stores the actual file contents; `.dvc` files store only metadata in Git