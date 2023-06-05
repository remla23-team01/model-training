import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

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
        X, y = np.load("data/X.npy"), np.load("data/y.npy")
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, random_state=0
        )
        model = joblib.load("ml_models/c2_Classifier_Sentiment_Model")
        assert sum(model.predict(X_test) == y_test) / len(y_test) > 0.5

    def test_non_determinism_robustness(self):
        """
        Test non-determinism robustness.
        """
        accuracy = 0.9
        X, y = np.load("data/X.npy"), np.load("data/y.npy")

        for i in range(10):
            model = train_model(X, y, random_state=i)
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.20, random_state=0
            )
            new_accuracy = sum(model.predict(X_test) == y_test) / len(y_test)
            assert accuracy - new_accuracy < 0.2
