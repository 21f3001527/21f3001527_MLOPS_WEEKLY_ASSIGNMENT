import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("MLSecOps_IRIS_Poisoning")

def train_and_log(csv_path, poison_level):
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    with mlflow.start_run(run_name=f"poison_{poison_level}pct"):
        mlflow.log_param("poison_level_pct", poison_level)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples",  len(X_test))
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)
        mlflow.sklearn.log_model(model, "model")
        report = classification_report(y_test, y_pred, target_names=['Setosa','Versicolor','Virginica'])
        rpath = f"data/report_{poison_level}pct.txt"
        open(rpath,'w').write(f"Poison: {poison_level}%\n\n" + report)
        mlflow.log_artifact(rpath)
        print(f"Poison {poison_level:>2}% | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

for path, lvl in [("data/iris_clean.csv",0),("data/iris_poisoned_5.csv",5),("data/iris_poisoned_10.csv",10),("data/iris_poisoned_50.csv",50)]:
    train_and_log(path, lvl)

print("\nAll experiments logged ✅")
