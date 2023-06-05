import os

import joblib
import numpy as np

from src.preprocess import main
from src.train_model import train_model


class TestModelDevelopment:
    def test_model_development(self):
        """
        Tests the model performs better than randomly selecting positive or negative.
        """
        if not os.path.exists("data/X.npy") or not os.path.exists(
            "data/y.npy"
        ):
            main()
        # TODO: use testing data instead of training data
        # TODO: load classifier weights from dvc (?)
        X, y = np.load("data/X.npy"), np.load("data/y.npy")
        model = joblib.load("ml_models/c2_Classifier_Sentiment_Model")
        assert sum(model.predict(X) == y) / len(y) > 0.5

    def test_non_determinism_robustness(self):
        """
        Test non-determinism robustness.
        """
        accuracy = 0.9
        X, y = np.load("data/X.npy"), np.load("data/y.npy")

        for i in range(10):
            model = train_model(X, y, random_state=i)
            assert sum(model.predict(X) == y) / len(y) > accuracy
