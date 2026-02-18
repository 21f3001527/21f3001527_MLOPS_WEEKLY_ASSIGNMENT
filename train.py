from google.cloud import storage
import pandas as pd
import joblib
import datetime
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BUCKET_NAME = "iris-training-data-21f3001527"

def train():

    print("Starting training...")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # 1️⃣ Download data from GCS
    blob = bucket.blob("data/iris.csv")
    blob.download_to_filename("iris.csv")

    data = pd.read_csv("iris.csv")

    # If column names are unknown, use this:
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    os.makedirs("artifacts", exist_ok=True)

    model_path = "artifacts/model.joblib"
    metrics_path = "artifacts/metrics.txt"

    joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {acc}")

    # 2️⃣ Upload artifacts to GCS output folder
    bucket.blob(f"output/{timestamp}/model.joblib").upload_from_filename(model_path)
    bucket.blob(f"output/{timestamp}/metrics.txt").upload_from_filename(metrics_path)

    print(f"Artifacts uploaded to output/{timestamp}/")


if __name__ == "__main__":
    train()
    train()   # run twice

