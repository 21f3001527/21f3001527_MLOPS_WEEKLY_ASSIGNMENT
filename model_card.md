# Model Card — IRIS Flower Classifier
**Assignment:** Week 9 — Explainability, Fairness & Drift Detection
**Roll Number:** 21f3001527
**Branch:** week_09

---

## 1. Model Overview

| Field | Details |
|---|---|
| **Model Type** | Random Forest Classifier (scikit-learn) |
| **Task** | Multi-class classification (Setosa / Versicolor / Virginica) |
| **Input Features** | sepal length, sepal width, petal length, petal width |
| **Output** | Predicted IRIS species (0, 1, or 2) |
| **Framework** | scikit-learn |

---

## 2. Intended Use
- Classify IRIS flower species from four morphological measurements.
- Intended for students and ML practitioners exploring explainability, fairness, and drift monitoring.
- Not intended for real-world production use.

---

## 3. Training Data
- **Dataset:** UCI IRIS Dataset (150 samples, 3 balanced classes)
- **Train/Test Split:** 120 training / 30 test samples
- **Features Used:** 4 original morphological features only
- **Sensitive Attribute:** location (randomly assigned 0 or 1) — for fairness auditing only, not used in training

---

## 4. Performance Metrics

### Overall
| Metric | Score |
|---|---|
| Accuracy | 0.9000 |
| Precision | 0.9024 |
| Recall | 0.9000 |

### By Location Group
| Location | Accuracy | Precision | Recall |
|---|---|---|---|
| Group 0 | 0.8571 | 0.8889 | 0.9048 |
| Group 1 | 0.9375 | 0.9167 | 0.9444 |
| Gap | 0.0804 | 0.0278 | 0.0397 |

---

## 5. Explainability (SHAP)
- Petal length and petal width are the strongest predictors for Virginica.
- High petal values (red dots on right) strongly push model toward Virginica.
- Sepal features have lower and more variable impact.

---

## 6. Drift Monitoring
| Feature | Drifted? |
|---|---|
| sepal length | No |
| sepal width | No |
| petal length | Yes (KS=0.4167) |
| petal width | Yes (KS=0.4250) |

Retraining triggered when KS p-value < 0.05.

---

## 7. Known Limitations
- Small dataset (150 samples) — high metric variance.
- Location attribute is randomly assigned — fairness analysis is for skill-building only.
- No hyperparameter tuning applied.

---

## 8. Fairness Considerations
- No sensitive attribute included in training features.
- Fairness audited using Fairlearn MetricFrame.
- Observed gap (~8% accuracy) attributed to random test-split variation.
