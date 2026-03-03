import joblib
import numpy as np
import unittest

class TestIrisModel(unittest.TestCase):

    def setUp(self):
        self.model = joblib.load("model.pkl")

    def test_model_loads(self):
        self.assertIsNotNone(self.model)

    def test_single_prediction(self):
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        pred = self.model.predict(sample)
        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], [0, 1, 2])

    def test_batch_prediction(self):
        samples = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 2.9, 4.3, 1.3],
            [7.7, 3.8, 6.7, 2.2]
        ])
        preds = self.model.predict(samples)
        self.assertEqual(len(preds), 3)

if __name__ == "__main__":
    unittest.main()
