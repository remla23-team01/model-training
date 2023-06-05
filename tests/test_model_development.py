import os

import joblib
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.preprocess import main
from src.train_model import train_model


@pytest.fixture
def X():
    """
    Load X data
    """
    return np.load("data/X.npy")


@pytest.fixture
def y():
    """
    Load y data
    """
    return np.load("data/y.npy")


class TestModelDevelopment:
    def test_model_development(self, X, y):
        """
        Tests the model performs better than randomly selecting positive or negative.
        """
        if not os.path.exists("data/X.npy") or not os.path.exists(
            "data/y.npy"
        ):
            main()
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, random_state=0
        )
        model = joblib.load("ml_models/c2_Classifier_Sentiment_Model")
        assert sum(model.predict(X_test) == y_test) / len(y_test) > 0.5

    def test_non_determinism_robustness(self, X, y):
        """
        Test non-determinism robustness.
        """
        accuracy = 0.9

        for i in range(10):
            model = train_model(X, y, random_state=i)
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.20, random_state=0
            )
            new_accuracy = sum(model.predict(X_test) == y_test) / len(y_test)
            assert abs(accuracy - new_accuracy) < 0.2

    def test_data_slices(self, X, y):
        """
        Test data slices.
        """

        model = joblib.load("ml_models/c2_Classifier_Sentiment_Model")

        # data splice

        pos = [X[i] for i in range(len(y)) if y[i]]
        neg = [X[i] for i in range(len(y)) if not y[i]]

        pos_acc = sum(model.predict(pos) == 1) / len(pos)
        neg_acc = sum(model.predict(neg) == 0) / len(neg)

        # compare accuracies
        assert abs(pos_acc - neg_acc) < 0.2
