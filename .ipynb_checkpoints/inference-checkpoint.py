from google.cloud import storage
import pandas as pd
import joblib
import datetime
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BUCKET_NAME = "iris-training-data-21f3001527"

def run_pipeline():

    print("Starting run...")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob("data/iris.csv")
    blob.download_to_filename("iris.csv")

    data = pd.read_csv("iris.csv")

    X = data[['sepal_length','sepal_width','petal_length','petal_width']]
    y = data['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, "artifacts/model.joblib")

    with open("artifacts/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}")

    bucket.blob(f"output/{timestamp}/model.joblib").upload_from_filename("artifacts/model.joblib")
    bucket.blob(f"output/{timestamp}/metrics.txt").upload_from_filename("artifacts/metrics.txt")

    print("Uploaded to output folder")

    sample = X_test.iloc[[0]]
    prediction = model.predict(sample)
    print("Sample Prediction:", prediction)


if __name__ == "__main__":
    run_pipeline()
    run_pipeline()
