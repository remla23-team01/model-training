import argparse
import numpy as np
import pandas as pd
import re
import pickle
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus = []

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    return corpus

def preprocess(dataset):
    """
    Preprocess dataset and return X and the count vectorizer object
    args:
        dataset: pandas dataframe
    """
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(dataset).toarray()
    return X, cv


def save_preprocessing(cv, path):
    """
    save preprocessing object to a specified path
    args:
        cv: sklearn preprocces object
        path: str
    """
    pickle.dump(cv, open(path, "wb"))

def save_preprocessed_data(X, y, folder):
    """
    save numpy arrays of preproccessed data to a specified folder
    args:
        X: numpy array
        y: numpy array
        folder: str
    """
    np.save('{}/X.npy'.format(folder), X) # save X
    np.save('{}/y.npy'.format(folder), y) # save y

def main():
    print("Loading dataset...")
    dataset = get_dataset('data/a1_RestaurantReviews_HistoricDump.csv')
    print("Dataset loaded!")
    
    print(dataset)
    
    print("Preprocessing dataset...")
    no_stopwords = remove_stopwords(dataset)
    X, cv = preprocess(no_stopwords)
    print("Dataset preprocessed, removed stopwords and used a count vectorizer")
    
    print("saving preprocessing step...")
    save_preprocessing(cv, "ml_models/preproccesing_object.pkl")
    save_preprocessed_data(X, dataset.iloc[:, -1].values, "data")
    print("succesfully saved preprocessing step to 'ml_models/preproccesing_object.pkl'\n", 
          "saved X and y numpy arrays to folder '/data'")
    
if __name__ == "__main__":
    main()