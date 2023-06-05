import pytest

from src.preprocess import get_dataset


@pytest.fixture
def dataset():
    yield get_dataset("data/a1_RestaurantReviews_HistoricDump.tsv")


class TestFeaturesAndData:
    def test_feature_distribution(self, dataset):
        """
        Test that the distributions of each feature match your expectations:

        Tests that the ratio between positive and negative training data is not too large.
        """
        label_count = dataset.groupby("Liked").count()
        negative_label_count = label_count.iloc[0, 0]
        positive_label_count = label_count.iloc[1, 0]
        threshold = 0.2
        assert 1 - (negative_label_count / positive_label_count) < threshold

    def test_relationship_between_feature_and_target(self, dataset):
        """
        Test the relationship between each feature and the target.
        """

        # check correlation between features and target
        dataset["LikedStr"] = dataset["Liked"].apply(lambda x: str(x))
        correlation = dataset.Review.str.get_dummies().corrwith(
            dataset.Liked / dataset.Liked.max()
        )
        # dataset has weak or negligible linear relationship between the variables
        correlation_threshold = 0.06
        for corr in correlation:
            assert abs(corr) < correlation_threshold

    def test_feature_memory_usage(self, dataset):
        """
        Test the cost of each feature - memory usage.
        """
        memory_usage = dataset.memory_usage(deep=True)
        # max memory usage is 200000 bytes
        max_memory_usage = 200000
        assert memory_usage["Review"] < max_memory_usage
