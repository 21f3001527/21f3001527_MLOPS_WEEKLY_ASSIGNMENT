import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

with open("week9_data.pkl", "rb") as f:
    d = pickle.load(f)

X_train       = d["X_train"]
feature_names = list(d["feature_names"])

np.random.seed(99)
X_prod = X_train.copy()
X_prod["petal length (cm)"] = X_prod["petal length (cm)"] + 1.5
X_prod["petal width (cm)"]  = X_prod["petal width (cm)"]  + 0.8

print("=" * 55)
print("DRIFT DETECTION — KS Test Results")
print("=" * 55)
print(f"{'Feature':<25} {'KS Stat':>8} {'p-value':>10} {'Drifted?':>10}")
print("-" * 55)

drift_results = {}
for feature in feature_names:
    ks_stat, p_value = stats.ks_2samp(
        X_train[feature].values,
        X_prod[feature].values
    )
    drifted = "YES" if p_value < 0.05 else "NO"
    drift_results[feature] = drifted
    print(f"{feature:<25} {ks_stat:>8.4f} {p_value:>10.4f} {drifted:>10}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, feature in enumerate(feature_names):
    axes[i].hist(X_train[feature], bins=20, alpha=0.6,
                 color="steelblue", label="Training")
    axes[i].hist(X_prod[feature],  bins=20, alpha=0.6,
                 color="tomato",   label="Production")
    axes[i].set_title(f"{feature} — {drift_results[feature]}", fontsize=11)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
    axes[i].legend()

plt.suptitle("Feature Distribution: Training vs Production", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("drift_detection.png", dpi=150, bbox_inches="tight")
print("\nSaved: drift_detection.png")

print("""
INTERPRETATION:
- petal length & petal width shifted -> drift detected
- sepal features unchanged -> no drift
- A deployed model trained on original data will silently
  degrade because its decision boundaries no longer match
  the new production distribution
- Action: retrain when KS p-value < 0.05
""")

print("✅ Task 4 Complete")
