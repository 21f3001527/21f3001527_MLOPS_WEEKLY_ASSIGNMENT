import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

mlflow.set_tracking_uri("./mlruns")
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("MLSecOps_IRIS_Poisoning")
runs = client.search_runs(exp.experiment_id, order_by=["params.poison_level_pct ASC"])

data = []
for r in runs:
    data.append({
        "poison_level": int(r.data.params["poison_level_pct"]),
        "accuracy":     r.data.metrics["accuracy"],
        "precision":    r.data.metrics["precision"],
        "recall":       r.data.metrics["recall"],
        "f1_score":     r.data.metrics["f1_score"],
    })

df = pd.DataFrame(data).sort_values("poison_level").reset_index(drop=True)
print("\n=== RESULTS ==="); print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(12,6))
x = np.arange(len(df)); w = 0.2
ax.bar(x-1.5*w, df['accuracy'],  w, label='Accuracy',  color='steelblue')
ax.bar(x-0.5*w, df['precision'], w, label='Precision', color='darkorange')
ax.bar(x+0.5*w, df['recall'],    w, label='Recall',    color='green')
ax.bar(x+1.5*w, df['f1_score'],  w, label='F1 Score',  color='red')
ax.set_xticks(x); ax.set_xticklabels([f"{p}%" for p in df['poison_level']])
ax.set_ylim(0,1.1); ax.legend(); ax.set_title('Performance vs Poison Level')
ax.axhline(0.333, color='gray', linestyle='--')
plt.tight_layout(); plt.savefig('data/results_chart.png', dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(df.set_index('poison_level')[['accuracy','precision','recall','f1_score']],
            annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.3, vmax=1.0, ax=ax)
ax.set_title('Metric Heatmap'); plt.tight_layout()
plt.savefig('data/heatmap.png', dpi=150); plt.close()
print("Charts saved ✅")
