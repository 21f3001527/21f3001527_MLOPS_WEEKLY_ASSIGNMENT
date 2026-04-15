import pickle
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("week9_data.pkl", "rb") as f:
    d = pickle.load(f)

clf           = d["clf"]
X_test        = d["X_test"]
feature_names = list(d["feature_names"])
target_names  = list(d["target_names"])

# ── SHAP Explainer ────────────────────────────────────────────────
explainer  = shap.TreeExplainer(clf)
shap_obj   = explainer(X_test)        # new-style Explanation object

# shap_obj.values shape: (n_samples, n_features, n_classes)
shap_values = shap_obj.values         # (30, 4, 3)

print(f"SHAP values shape: {shap_values.shape}")
print(f"X_test shape     : {X_test.shape}")

# ── Plot all 3 classes ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

for i, name in enumerate(target_names):
    plt.sca(axes[i])
    shap.summary_plot(
        shap_values[:, :, i],   # slice for class i → (n_samples, n_features)
        X_test,
        feature_names=feature_names,
        show=False,
        plot_size=None,
        color_bar=(i == 2),
    )
    axes[i].set_title(f"SHAP — {name.capitalize()}", fontsize=13, fontweight="bold")

plt.suptitle("SHAP Summary — All Three IRIS Classes", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("shap_all_classes.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_all_classes.png")

# ── Virginica only ────────────────────────────────────────────────
plt.figure(figsize=(9, 6))
shap.summary_plot(
    shap_values[:, :, 2],       # class 2 = virginica
    X_test,
    feature_names=feature_names,
    show=False,
)
plt.title("SHAP Summary — Virginica", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_virginica.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_virginica.png")

print("""
VIRGINICA SHAP EXPLANATION:
- Right side (positive SHAP) = pushes model TOWARD virginica
- Left side (negative SHAP)  = pushes model AWAY from virginica
- Red dot  = high feature value for that sample
- Blue dot = low feature value for that sample
- petal length & petal width: red dots cluster on right
  meaning large petals strongly predict virginica
""")

print("✅ Task 3 Complete")
