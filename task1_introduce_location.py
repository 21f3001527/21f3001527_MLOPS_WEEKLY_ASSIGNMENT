import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

X['location'] = np.random.randint(0, 2, size=len(X))

print("=" * 55)
print("Dataset with Location Attribute (first 10 rows)")
print("=" * 55)
print(X.head(10))
print("\nLocation value counts:")
print(X['location'].value_counts())

sensitive_attr = X['location'].copy()
X_features = X.drop(columns=['location'])

X_train, X_test, y_train, y_test, loc_train, loc_test = train_test_split(
    X_features, y, sensitive_attr,
    test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "loc_train": loc_train,
    "loc_test": loc_test,
    "y_pred": y_pred,
    "clf": clf,
    "feature_names": iris.feature_names,
    "target_names": iris.target_names,
}

with open("week9_data.pkl", "wb") as f:
    pickle.dump(data, f)

print("\n✅ Task 1 Complete")
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Test samples     : {X_test.shape[0]}")
print(f"   Saved week9_data.pkl")
