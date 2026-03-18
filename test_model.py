import os
import unittest
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris

# Connect to MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8100"))

class TestIrisModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load model from MLflow Registry instead of local file"""
        client = mlflow.MlflowClient()
        
        # Fetch latest version from registry
        versions = client.search_model_versions("name='IRIS-classifier-rf'")
        latest_version = versions[-1].version
        
        print(f"Loading model version: {latest_version}")
        
        # Load model from MLflow registry
        cls.model = mlflow.sklearn.load_model(
            f"models:/IRIS-classifier-rf/{latest_version}"
        )
        
        # Load test data
        iris = load_iris()
        cls.X = iris.data
        cls.y = iris.target

    def test_model_loads(self):
        """Check model loads correctly from MLflow registry"""
        self.assertIsNotNone(self.model)
        print("✅ test_model_loads passed")

    def test_single_prediction(self):
        """Check single sample prediction returns valid class"""
        sample = self.X[0].reshape(1, -1)
        pred = self.model.predict(sample)
        self.assertIn(pred[0], [0, 1, 2])
        print("✅ test_single_prediction passed")

    def test_batch_prediction(self):
        """Check batch predictions return correct count"""
        samples = self.X[:3]
        preds = self.model.predict(samples)
        self.assertEqual(len(preds), 3)
        print("✅ test_batch_prediction passed")

    def test_prediction_accuracy(self):
        """Check model accuracy is above 90%"""
        from sklearn.metrics import accuracy_score
        preds = self.model.predict(self.X)
        acc = accuracy_score(self.y, preds)
        self.assertGreater(acc, 0.90)
        print(f"✅ test_prediction_accuracy passed — accuracy: {acc:.4f}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
