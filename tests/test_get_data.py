import pandas as pd
import pytest

from src.get_data import download_data, save_data


class TestGetData:
    def test_download_data(self):
        url = "data/a1_RestaurantReviews_HistoricDump.tsv"
        data = download_data(url)
        assert isinstance(data, pd.DataFrame)
