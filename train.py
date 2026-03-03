import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Save dataset locally (simulate raw data)
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y
df.to_csv("data.csv", index=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Save model
joblib.dump(model, "model.pkl")

# Save metrics
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")

print("Training complete. Accuracy:", acc)