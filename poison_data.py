import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

np.random.seed(42)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.to_csv('data/iris_clean.csv', index=False)
print("Clean dataset saved")

def poison_dataset(df, fraction, seed=42):
    np.random.seed(seed)
    poisoned = df.copy()
    n = int(len(poisoned) * fraction)
    indices = np.random.choice(len(poisoned), size=n, replace=False)
    for idx in indices:
        for col in iris.feature_names:
            poisoned.at[idx, col] = np.random.uniform(df[col].min(), df[col].max())
        poisoned.at[idx, 'label'] = np.random.randint(0, 3)
    print(f"Poisoned {n}/{len(df)} samples ({fraction*100:.0f}%)")
    return poisoned

poison_dataset(df, 0.05).to_csv('data/iris_poisoned_5.csv',  index=False)
poison_dataset(df, 0.10).to_csv('data/iris_poisoned_10.csv', index=False)
poison_dataset(df, 0.50).to_csv('data/iris_poisoned_50.csv', index=False)
print("All poisoned datasets saved ✅")
