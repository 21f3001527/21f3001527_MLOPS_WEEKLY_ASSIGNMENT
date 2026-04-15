import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame

with open("week9_data.pkl", "rb") as f:
    d = pickle.load(f)

y_test   = d["y_test"]
y_pred   = d["y_pred"]
loc_test = d["loc_test"]

metrics = {
    "accuracy":  lambda yt, yp: accuracy_score(yt, yp),
    "precision": lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0),
    "recall":    lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0),
}

mf = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=loc_test,
)

print("=" * 55)
print("OVERALL METRICS")
print("=" * 55)
print(mf.overall.round(4))

print("\n" + "=" * 55)
print("METRICS BY LOCATION GROUP")
print("=" * 55)
print(mf.by_group.round(4))

print("\n" + "=" * 55)
print("PERFORMANCE GAP (Group 1 minus Group 0)")
print("=" * 55)
gap = mf.by_group.loc[1] - mf.by_group.loc[0]
print(gap.round(4))

print("\n✅ Task 2 Complete")
