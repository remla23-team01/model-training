import numpy as np
import psutil
import pytest
from sklearn.model_selection import train_test_split

from src.models.train_model import train_model


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


class TestMonitoring:
    def test_model_staleness(self, X, y):
        """Test for model statelness."""

        # e.g. old data
        X_old, y_old = X[:300], y[:300]
        model_old = train_model(X_old, y_old)

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        old_acc = sum(model_old.predict(X_test) == y_test) / len(y_test)

        # e.g. new data arriving
        X_new, y_new = X[:800], y[:800]
        model_new = train_model(X_new, y_new)

        new_acc = sum(model_new.predict(X_test) == y_test) / len(y_test)

        assert new_acc > old_acc

    def test_ram_used_for_training(self, X, y):
        """Test for ram used for training."""
        pr = psutil.Process()
        ram_used_before = pr.memory_info().rss
        train_model(X, y)
        end_ram_used = pr.memory_info().rss
        # ram used for training is less than 1GB
        assert end_ram_used - ram_used_before < 1000000000
