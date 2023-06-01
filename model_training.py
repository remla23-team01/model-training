"""Module for training the sentiment model."""

import argparse
import re
import pickle
import numpy as np
import pandas as pd

import joblib

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from remla01_lib import mlSteps

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


def get_dataset(filename):
    """Gets the dataset from the given filename."""
    dataset = pd.read_csv(filename, delimiter='\t', quoting=3)
    return dataset


def remove_stopwords(dataset):
    """Removes stopwords from the dataset."""
    port_stemmer = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []

    for i in range(0, 900):
        review = mlSteps.remove_stopwords(dataset['Review'][i])
        corpus.append(review)

    return corpus


def preprocess(text):
    """Preprocesses the text."""
    count_vec = CountVectorizer(max_features=1420)
    transformation_arr = count_vec.fit_transform(text).toarray()
    return transformation_arr, count_vec


def save_preprocessing(count_vec):
    """Saves the preprocessing."""
    bow_path = 'ml_models/c1_BoW_Sentiment_Model.pkl'
    pickle.dump(count_vec, open(bow_path, "wb"))


def train_model(X, y):
    """Trains the model."""
    print('Training model...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'ml_models/c2_Classifier_Sentiment_Model')

    y_pred = classifier.predict(X_test)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print(confusion_mat)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


def run_preprocessing():
    """Runs the preprocessing."""
    print('Running preprocessing...')
    dataset = get_dataset('data/a1_RestaurantReviews_HistoricDump.tsv')
    no_stopwords = remove_stopwords(dataset)
    X, count_vec = preprocess(no_stopwords)
    y = dataset.iloc[:, -1].values
    np.save('data/X.npy', X)
    np.save('data/y.npy', y)
    save_preprocessing(count_vec)


def main():
    """Main function."""
    if rerun_preprocessing:
        run_preprocessing()
    try:
        X = np.load('data/X.npy')
        y = np.load('data/y.npy')
        train_model(X, y)
    except FileNotFoundError:
        print('Something went wrong when reading X and/or y or during training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_preprocessing", action='store_true', help="Add to run preprocessing")
    args = parser.parse_args()

    rerun_preprocessing = args.run_preprocessing

    main()
