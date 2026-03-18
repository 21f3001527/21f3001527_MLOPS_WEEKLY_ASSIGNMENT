import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from mlflow.models import infer_signature
import mlflow
import mlflow.sklearn
import itertools

# Connect to MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8100"))
mlflow.set_experiment("MLflow with IRIS Datset")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Save dataset
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y
df.to_csv("data.csv", index=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning — vary 2 params (Task 1)
n_estimators_list = [50, 100, 200]
max_depth_list = [3, 5]

best_acc = 0
best_run_id = None

for n_est, max_dep in itertools.product(n_estimators_list, max_depth_list):
    params = {
        "n_estimators": n_est,
        "max_depth": max_dep,
        "random_state": 42
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")

    with mlflow.start_run(run_name=f"rf-n{n_est}-d{max_dep}") as run:
        # Task 2 — Log params and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)
        mlflow.set_tag("Training Info", "Random Forest for IRIS data")

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="IRIS-classifier-rf"
        )

        if acc > best_acc:
            best_acc = acc
            best_run_id = run.info.run_id

    print(f"✅ n_estimators={n_est}, max_depth={max_dep} → accuracy={acc:.4f}")

print(f"\n🏆 Best run ID: {best_run_id} with accuracy: {best_acc:.4f}")
print("🎉 All runs logged to MLflow!")
