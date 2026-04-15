import pickle
import numpy as np

def test_model_loads():
    import joblib, os
    # Check that week9_data.pkl exists
    assert os.path.exists("week9_data.pkl"), "Data file missing"

def test_model_output():
    import joblib
    data = joblib.load("week9_data.pkl")
    assert data is not None
