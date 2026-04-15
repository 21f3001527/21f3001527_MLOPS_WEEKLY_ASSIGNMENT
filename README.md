# 21f3001527_MLOPS_WEEKLY_ASSIGNMENT

---

## Week 9 — Explainability, Fairness & Drift Detection

### Overview
- **Explainability** — using SHAP to explain model decisions
- **Fairness Auditing** — using Fairlearn to check performance across groups
- **Drift Monitoring** — using KS statistical tests to detect distribution shifts


### Repository Structure
```
21f3001527_MLOPS_WEEKLY_ASSIGNMENT/
├── task1_introduce_location.py   # Adds location sensitive attribute
├── task2_fairness_analysis.py    # Fairlearn MetricFrame fairness audit
├── task3_shap_explainability.py  # SHAP summary plots
├── task4_drift_detection.py      # KS test drift detection
├── model_card.md                 # Task 5 - Model Card
├── run_pipeline.sh               # Runs all tasks at once
├── week9_data.pkl                # Saved train/test data
├── shap_all_classes.png          # SHAP plot for all classes
├── shap_virginica.png            # SHAP plot for Virginica
├── drift_detection.png           # Drift detection plot
├── requirements.txt              # Python dependencies
└── README.md
```


### Task 1 — Introduce Location Attribute
- Added `location` column (randomly assigned 0 or 1) to IRIS dataset
- Model trained on original 4 features only
- Location used only as sensitive attribute for fairness auditing

### Task 2 — Fairness Analysis with Fairlearn
- Used Fairlearn `MetricFrame` with `location` as sensitive attribute
- Metrics disaggregated by location group (0 and 1)

| Location | Accuracy | Precision | Recall |
|---|---|---|---|
| Group 0 | 0.8571 | 0.8889 | 0.9048 |
| Group 1 | 0.9375 | 0.9167 | 0.9444 |
| Gap | 0.0804 | 0.0278 | 0.0397 |

> Gap is due to random test-split variation, not real bias.

### Task 3 — SHAP Explainability
- Generated SHAP summary plots for all 3 IRIS classes
- Virginica finding: petal length & petal width are strongest predictors
- Large petal values (red dots on right) strongly push model toward Virginica

### Task 4 — Drift Detection
- Simulated production data by shifting petal feature distributions
- Used Kolmogorov-Smirnov (KS) test to detect drift

| Feature | KS Stat | p-value | Drifted? |
|---|---|---|---|
| sepal length | 0.0000 | 1.0000 | No |
| sepal width | 0.0000 | 1.0000 | No |
| petal length | 0.4167 | 0.0000 | Yes |
| petal width | 0.4250 | 0.0000 | Yes |

> Retraining triggered when KS p-value < 0.05

### Task 5 — Model Card
- Written in `model_card.md`
- Covers intended use, training data, performance metrics, SHAP findings, drift results, known limitations and fairness considerations

### How to Run
```bash
# Run all tasks at once
bash run_pipeline.sh

# Or run individually
python task1_introduce_location.py
python task2_fairness_analysis.py
python task3_shap_explainability.py
python task4_drift_detection.py
```

### Tech Stack
- Python 3.9
- scikit-learn — Model training
- SHAP — Explainability
- Fairlearn — Fairness auditing
- SciPy — KS drift detection
- matplotlib — Plots

---


# python task1_introduce_location.py

# python task2_fairness_analysis.py

# python task3_shap_explainability.py

# eog shap_virginica.png 

# python task4_drift_detection.py

<!-- eog drift-detection.pmg -->
