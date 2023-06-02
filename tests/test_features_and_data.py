from src.preprocess import get_dataset


class TestFeaturesAndData:
    def test_feature_distribution(self):
        training_data = get_dataset(
            "data/a1_RestaurantReviews_HistoricDump.tsv"
        )
        label_count = training_data.groupby("Liked").count()
        negative_label_count = label_count.iloc[0, 0]
        positive_label_count = label_count.iloc[1, 0]
        assert 1 - (negative_label_count / positive_label_count) < 0.2
