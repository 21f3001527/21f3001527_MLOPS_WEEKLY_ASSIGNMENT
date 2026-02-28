import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

store = FeatureStore(repo_path="iris_feature_repo")

# Task 4 - Offline Training
print("Fetching from offline store (BigQuery)...")
training_df = store.get_historical_features(
    entity_df=pd.DataFrame({
        "iris_id": list(range(150)),
        "event_timestamp": pd.Timestamp("2024-01-01", tz="UTC")
    }),
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
        "iris_features:species",
    ],
).to_df()

print(training_df.head())
X = training_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = training_df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model Accuracy: {acc:.4f}")

# Task 5 - Online Inference
print("\nFetching from online store (SQLite)...")
online_features = store.get_online_features(
    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ],
    entity_rows=[{"iris_id": 0}, {"iris_id": 1}, {"iris_id": 2}],
).to_dict()

inference_df = pd.DataFrame({
    "sepal_length": online_features["sepal_length"],
    "sepal_width":  online_features["sepal_width"],
    "petal_length": online_features["petal_length"],
    "petal_width":  online_features["petal_width"],
})
predictions = model.predict(inference_df)
species_names = load_iris().target_names
print("\n✅ Inference Results:")
for iris_id, pred in zip([0,1,2], predictions):
    print(f"  iris_id={iris_id} → {species_names[pred]}")
