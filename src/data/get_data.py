"""Gets raw data from a url and saves it to a specified path."""
import pandas as pd


def download_data(url):
    """
    Download raw data from a url. Needs to be a url to the raw data
    args:
        url: str
    """
    print("Downloading data...")
    dataset = pd.read_csv(url, sep=",", header=0)
    print("Succesfully downloaded data!")
    return dataset


def save_data(data, path):
    """
    Save data to a specified path
    args:
        data: pandas dataframe
        path: str
    """
    print("Saving data...")
    data.to_csv(path, index=False)
    print(f"Succesfully saved data to {path}")


def get_data():
    """Main function to run script"""
    url = "https://raw.githubusercontent.com/remla23-team01/"
    url = (
       url + "model-training/main/data/a1_RestaurantReviews_HistoricDump.tsv"
    )
    # url = "data/a1_RestaurantReviews_HistoricDump.tsv"
    data = download_data(url)
    print(f"Preview of data:\n, {data.head()}")
    path = "data/raw/a1_RestaurantReviews_HistoricDump.csv"
    save_data(data, path)

if __name__ == "__main__":
    get_data()
