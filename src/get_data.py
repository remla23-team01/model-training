import pandas as pd

def download_data(url):
    """
    Download raw data from a url. Needs to be a url to the raw data
    args:
        url: str 
    """
    print('Downloading data...')
    dataset = pd.read_csv(url, delimiter='\t', quoting=3, header=0)
    print('Succesfully downloaded data!')
    return dataset

def save_data(data, path):
    """
    Save data to a specified path
    args:
        data: pandas dataframe
        path: str
    """
    print('Saving data...')
    data.to_csv(path, index=False)
    print('Succesfully saved data to "{}"'.format(path))

def main():
    url = "https://raw.githubusercontent.com/remla23-team01/model-training/main/data/a1_RestaurantReviews_HistoricDump.tsv"
    data = download_data(url)
    print("Preview of data:\n", data.head(), data.shape)
    
    path = "data/a1_RestaurantReviews_HistoricDump.csv"
    save_data(data, path)

if __name__ == '__main__':
    main()