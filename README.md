# 21f3001527 - MLOps Weekly Assignment

## Week 3: Feast Feature Store Integration

### Overview
Integrated the Feast Feature Store into the IRIS classification pipeline to eliminate training-serving skew by providing a single source of truth for features during both training and inference.

---

21f3001527_MLOPS_WEEKLY_ASSIGNMENT/
├── iris_feature_repo/
│   ├── feature_store.yaml
│   ├── iris_features.py
│   └── data/
│       ├── iris.parquet
│       ├── registry.db
│       └── online_store.db
├── train_predict.py
└── README.md

---

## Tasks Completed

### Task 1 - Initialize Feast Feature Repository
- Initialized Feast repo with local SQLite backend
- Configured feature_store.yaml with offline (BigQuery) and online (SQLite) stores

### Task 2 - Define Entities, Data Sources and Feature Views
- Defined iris_id as the entity to uniquely identify each iris sample
- Created a BigQuerySource pointing to feast_iris.iris_data table
- Defined iris_features FeatureView with sepal_length, sepal_width, petal_length, petal_width and species

### Task 3 - Apply and Materialize Features
- Registered all definitions using feast apply
- Materialized 150 feature rows into online store using feast materialize
- Verified materialization by querying the online store successfully

### Task 4 - Fetch Features for Training (Offline Store)
- Used get_historical_features() to fetch all 150 samples from BigQuery offline store
- Trained a RandomForestClassifier entirely on Feast-retrieved features
- Model Accuracy: 96.67%

### Task 5 - Fetch Features for Inference (Online Store)
- Used get_online_features() for real-time low-latency feature retrieval
- Predicted species for iris_id 0, 1, 2
- Confirmed predictions are consistent with raw data

### Task 6 - BigQuery Backend (Optional)
- Replaced local FileSource with BigQuery offline store
- Uploaded IRIS dataset to BigQuery table: feast_iris.iris_data
- Configured feature_store.yaml with provider: gcp and offline_store: bigquery
- Online store remains SQLite for low-latency inference

---

## Key Concepts Learned
- Feature stores eliminate training-serving skew
- Offline store (BigQuery) is used for batch/historical features during training
- Online store (SQLite) is used for low-latency features during inference
- Materialization moves features from offline store to online store

---

## How to Run

### Install dependencies
pip install feast scikit-learn pandas feast[gcp]

### Upload data to BigQuery
python3 iris_upload_bq.py

### Apply Feast definitions
cd iris_feature_repo
feast apply

### Materialize features into online store
feast materialize 2024-01-01T00:00:00 2026-02-28T23:59:59

### Train and Predict
cd ..
python3 train_predict.py

---

## Tech Stack
- Feast: Feature Store
- BigQuery: Offline Store backend
- SQLite: Online Store backend
- scikit-learn: Model training
- Python 3.10
- GCP: Google Cloud Platform
