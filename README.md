# Week 08 — Integrating MLSecOps into the IRIS Pipeline

## Overview
This week explores ML security threat vectors and simulates data poisoning attacks on the IRIS dataset at varying severity levels (5%, 10%, 50%), measuring the impact on model performance using MLflow.

---

## Repository Structure
```
21f3001527_MLOPS_WEEKLY_ASSIGNMENT/
├── .github/
│   └── workflows/
│       └── mlsecops_week08.yml   # GitHub Actions CI pipeline
├── data/
│   ├── iris_clean.csv            # Original clean dataset
│   ├── iris_poisoned_5.csv       # 5% corrupted dataset
│   ├── iris_poisoned_10.csv      # 10% corrupted dataset
│   ├── iris_poisoned_50.csv      # 50% corrupted dataset
│   ├── results_chart.png         # Performance comparison bar chart
│   ├── heatmap.png               # Metric heatmap across poison levels
│   ├── report_0pct.txt           # Classification report - clean
│   ├── report_5pct.txt           # Classification report - 5% poisoned
│   ├── report_10pct.txt          # Classification report - 10% poisoned
│   └── report_50pct.txt          # Classification report - 50% poisoned
├── poison_data.py                # Task 2 - generates poisoned datasets
├── train_log.py                  # Task 3 - trains models & logs to MLflow
└── analyze.py                    # Task 4 - generates comparison charts
```

---

## Tasks

### Task 1 — ML Threat Vectors
Four key security threat vectors in ML systems:

| Threat | Pipeline Stage | Description |
|--------|---------------|-------------|
| **Data Poisoning** | Data Ingestion | Attacker corrupts training data to degrade accuracy or introduce targeted misclassifications |
| **Adversarial Examples** | Inference | Crafted inputs with small perturbations that cause the model to misclassify |
| **Model Extraction** | Deployment | Adversary queries the API repeatedly to clone/steal the model |
| **Prompt Injection** | Inference API | Malicious instructions embedded in user input to override LLM behavior |

---

### Task 2 — Poison the IRIS Dataset
**Script:** `poison_data.py`

Generates 3 poisoned variants of the IRIS dataset by replacing a percentage of samples with random feature values and random labels:

```bash
python poison_data.py
```

**Output:**
```
Clean dataset saved
Poisoned 7/150 samples (5%)
Poisoned 15/150 samples (10%)
Poisoned 75/150 samples (50%)
All poisoned datasets saved ✅
```

**Poisoning Method:** For each corrupted sample, all four features (sepal length, sepal width, petal length, petal width) are replaced with random values within the observed range, and a random class label (0, 1, or 2) is assigned.

---

### Task 3 — Train & Log Experiments in MLflow
**Script:** `train_log.py`

Trains a Random Forest classifier on each dataset variant and logs all runs to MLflow:

```bash
python train_log.py
```

**MLflow Experiment:** `MLSecOps_IRIS_Poisoning`  
**Tracked Parameters:** `poison_level_pct`, `train_samples`, `test_samples`  
**Tracked Metrics:** `accuracy`, `precision`, `recall`, `f1_score`

Launch MLflow UI to compare runs:
```bash
mlflow ui --port 5000
```

---

### Task 4 — Results & Analysis

| Poison % | Accuracy | Precision | Recall | F1 Score |
|----------|----------|-----------|--------|----------|
| 0%       | 0.9333   | 0.9333    | 0.9333 | 0.9333   |
| 5%       | 0.9333   | 0.9444    | 0.9333 | 0.9346   |
| 10%      | 0.9333   | 0.9364    | 0.9333 | 0.9331   |
| 50%      | 0.6000   | 0.6342    | 0.6000 | 0.6076   |

**Key Findings:**
- At **5% and 10%** poisoning — no measurable degradation. IRIS has well-separated classes that absorb small amounts of noise
- At **50%** poisoning — sharp drop to 60% accuracy (~33 point F1 drop)
- At 50% the model is still above random chance (33% for 3-class) but is unreliable for production use
- All metrics degrade together at 50% — no single metric is more resilient

---

### Task 5 — Mitigation Strategies

**Detecting & Mitigating Data Poisoning:**
- **Statistical Validation** — check feature distributions before training; flag samples outside expected ranges
- **Schema Enforcement** — reject values outside biological bounds (e.g. sepal length > 10cm)
- **Anomaly Detection** — use isolation forests or Z-score checks to detect outlier samples
- **Data Provenance Tracking** — track the origin of every training sample to identify compromised sources

**Data Quantity vs Data Quality:**
- Adding more poisoned data **amplifies** the problem — it does not help
- At 50% corruption the model effectively learns noise, not patterns
- The priority must be **cleaning contaminated samples first**, then assessing if remaining clean data is sufficient
- In production: enforce minimum clean data ratio thresholds before allowing a training run to proceed

---

## GitHub Actions CI Pipeline

The workflow automatically runs on every push to `week_08` branch:

```
✅ Checkout Code
✅ Install Dependencies
✅ Run Data Poisoning
✅ Train & Log to MLflow
✅ Analyze Results
```

---

## How to Run Locally

```bash
# Install dependencies
pip install mlflow scikit-learn pandas numpy matplotlib seaborn

# Run full pipeline
python poison_data.py
python train_log.py
python analyze.py


```

---

<!-- ## mlflow ui
mlflow ui --host 0.0.0.0 --port 5000 \
  --backend-store-uri ./mlruns \
  --allowed-hosts 4ae250eed067c0a6-dot-us-central1.notebooks.googleusercontent.com \
  --cors-allowed-origins https://4ae250eed067c0a6-dot-us-central1.notebooks.googleusercontent.com

  https://4ae250eed067c0a6-dot-us-central1.notebooks.googleusercontent.com/proxy/5000/ -->