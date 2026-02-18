# 21f3001527_MLOPS_WEEKLY_ASSIGNMENT

# Iris ML Training Pipeline with GCS Integration

## Project Overview

This project implements an end-to-end Machine Learning training and inference pipeline using the Iris dataset.

The pipeline performs the following steps:

* Downloads dataset from Google Cloud Storage (GCS)
* Trains a Decision Tree Classifier
* Evaluates model performance using accuracy
* Saves model and metrics locally
* Uploads artifacts to GCS with timestamp-based versioning
* Supports inference using the saved model

This project demonstrates basic MLOps practices including cloud storage integration, artifact versioning, and reproducible training.

---

## Model Details

* Algorithm: Decision Tree Classifier
* Library: scikit-learn
* Hyperparameter:

  * max_depth = 3
* Train/Test Split Configuration:

  * test_size = 0.4
  * stratify = y
  * random_state = 42
* Evaluation Metric: Accuracy Score

Typical Accuracy Achieved:

0.95 – 0.98

---

## Project Structure

```
.
├── train.py
├── inference.py
├── requirements.txt
├── artifacts/
│   ├── model.joblib
│   └── metrics.txt
└── README.md
```

---

## Google Cloud Storage (GCS) Structure

Bucket Name:

```
iris-training-data-21f3001527
```

Inside Bucket:

```
data/iris.csv
output/<timestamp>/model.joblib
output/<timestamp>/metrics.txt
```

Each training run creates a new timestamped folder inside the output directory.

Example:

```
output/20260218_160412/
```

This ensures artifact versioning for every training execution.

---

## Training Pipeline Workflow

1. Connects to Google Cloud Storage.
2. Downloads dataset from:

   ```
   gs://iris-training-data-21f3001527/data/iris.csv
   ```
3. Splits dataset into training and testing sets.
4. Trains:

   ```
   DecisionTreeClassifier(max_depth=3)
   ```
5. Evaluates accuracy.
6. Saves artifacts locally:

   ```
   artifacts/model.joblib
   artifacts/metrics.txt
   ```
7. Uploads artifacts to GCS:

   ```
   output/<timestamp>/model.joblib
   output/<timestamp>/metrics.txt
   ```

---

## How to Run Training

```
python train.py
```

Expected Output:

```
Starting training...
Accuracy: 0.95
Artifacts uploaded to output/<timestamp>/
```

Note: If train() is called twice in the script, it will run twice and create two separate timestamp folders.

---

## Inference

The inference script:

* Loads the trained model.joblib
* Takes input features
* Predicts Iris class

Run inference using:

```
python inference.py
```

---

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

Main Libraries Used:

* google-cloud-storage
* pandas
* scikit-learn
* joblib

---

## Key Features

* Google Cloud Storage integration
* Automatic artifact upload
* Timestamp-based model versioning
* Model persistence using joblib
* Reproducible training setup
* Separate training and inference scripts
