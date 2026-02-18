from google.cloud import storage
import joblib
import pandas as pd
import os

BUCKET_NAME = "iris-training-data-21f3001527"

def inference():

    print("Starting inference...")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # 1️⃣ Get latest timestamp folder
    blobs = list(bucket.list_blobs(prefix="output/"))
    folders = list(set(blob.name.split('/')[1] for blob in blobs if blob.name.count('/') > 1))
    latest_folder = sorted(folders)[-1]

    print("Using model from:", latest_folder)

    os.makedirs("artifacts", exist_ok=True)

    # 2️⃣ Download model
    bucket.blob(f"output/{latest_folder}/model.joblib").download_to_filename("artifacts/model.joblib")

    model = joblib.load("artifacts/model.joblib")

    # 3️⃣ Download data again for evaluation
    bucket.blob("data/iris.csv").download_to_filename("iris.csv")
    data = pd.read_csv("iris.csv")

    X = data.iloc[:, :-1]

    sample = X.iloc[[0]]
    prediction = model.predict(sample)

    print("Sample Prediction:", prediction)


if __name__ == "__main__":
    inference()
    inference()   # run twice
