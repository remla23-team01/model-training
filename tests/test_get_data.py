import os

import pandas as pd

from src.get_data import download_data, get_data, save_data


class TestGetData:
    def test_download_data(self):
        url = "data/a1_RestaurantReviews_HistoricDump.tsv"
        data = download_data(url)
        assert isinstance(data, pd.DataFrame)

    def test_save_data(self):
        url = "data/a1_RestaurantReviews_HistoricDump.tsv"
        data = download_data(url)
        saving_path = "data/a1_RestaurantReviews_HistoricDump.csv"
        save_data(data, saving_path)
        # check if the file exists
        assert os.path.isfile(saving_path)

    def test_get_data(self):
        get_data()
        saving_path = "data/a1_RestaurantReviews_HistoricDump.csv"
        assert os.path.isfile(saving_path)
        data = pd.read_csv(saving_path)
        assert not data.empty
