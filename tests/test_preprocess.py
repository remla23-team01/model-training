import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from src.features.preprocess import get_dataset, preprocess, remove_stopwords


@pytest.fixture
def dataset():
    yield get_dataset("data/a1_RestaurantReviews_HistoricDump.tsv")


@pytest.fixture
def mock_dataset():
    # Create a sample dataset for testing
    df = pd.DataFrame(
        {
            "Review": [
                "This is a good movie",
                "I did not like the book",
                "The food was not tasty",
            ]
        }
    )
    return df


class TestPreprocess:
    def test_get_dataset(self, dataset):
        assert isinstance(dataset, pd.DataFrame)
        assert not dataset.empty
        assert dataset.shape == (900, 2)

    def test_remove_stopwords(self, mock_dataset):
        # Call the method to remove stopwords
        corpus = remove_stopwords(mock_dataset)

        # Assert the expected output
        expected_corpus = ["good movi", "not like book", "food not tasti"]
        assert corpus == expected_corpus

    def test_preprocess(self, mock_dataset):
        # Call the method to preprocess the data
        transformed_vec, count_vec = preprocess(mock_dataset)

        # Assert the expected output
        assert transformed_vec == [[1]]
        assert isinstance(count_vec, CountVectorizer)
