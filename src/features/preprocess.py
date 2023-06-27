"""Module to preprocess the dataset and save the preprocessed data and preprocessing object"""

import pickle

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from remla01_lib import mlSteps
from sklearn.feature_extraction.text import CountVectorizer


def get_dataset(path):
    """
    Load dataset from a specified path
    args:
        path: str
    """
    dataset = pd.read_csv(path)
    return dataset


def remove_stopwords(dataset):
    """
    Remove stopwords from dataset
    args:
        dataset: pandas dataframe
    """

    corpus = []

    for i in range(dataset.shape[0]):
        review = mlSteps.remove_stopwords(dataset["Review"][i])
        corpus.append(review)

    return corpus


def preprocess_dataset(dataset):
    """
    Preprocess dataset and return X and the count vectorizer object
    args:
        dataset: pandas dataframe
    """
    count_vec = CountVectorizer(max_features=1420)
    transformed_vec = count_vec.fit_transform(dataset).toarray()
    return transformed_vec, count_vec


def save_preprocessing(count_vec, path):
    """
    save preprocessing object to a specified path
    args:
        count_vec: sklearn preprocces object
        path: str
    """
    pickle.dump(count_vec, open(path, "wb"))


def save_preprocessed_data(X, y, folder):
    """
    save numpy arrays of preproccessed data to a specified folder
    args:
        X: numpy array
        y: numpy array
        folder: str
    """
    np.save(f"{folder}/X.npy", X)  # save X
    np.save(f"{folder}/y.npy", y)  # save y


def main():
    """Main function to run script"""
    print("Loading dataset...")
    dataset = get_dataset("data/raw/a1_RestaurantReviews_HistoricDump.csv")
    print("Dataset loaded!")
    print(dataset)
    print("Preprocessing dataset...")
    no_stopwords = remove_stopwords(dataset)
    X, cv = preprocess_dataset(no_stopwords)
    print("Dataset preprocessed, removed stopwords and used a count vectorizer")
    print("saving preprocessing step...")
    save_preprocessing(cv, "ml_models/preproccesing_object.pkl")
    save_preprocessed_data(X, dataset.iloc[:, -1].values, "data/processed")
    print(
        "succesfully saved preprocessing step to 'ml_models/preproccesing_object.pkl'\n",
        "saved X and y numpy arrays to folder '/data/processed'",
    )


if __name__ == "__main__":
    main()
