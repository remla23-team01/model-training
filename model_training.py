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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


def get_dataset(filename):
    dataset = pd.read_csv(filename, delimiter='\t', quoting=3)
    return dataset


def remove_stopwords(dataset):
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


def preprocess(text):
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(text).toarray()
    return X, cv


def save_preprocessing(cv):
    bow_path = 'data/c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))


def train_model(X, y):
    print('Training model...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'data/c2_Classifier_Sentiment_Model')

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


def run_preprocessing():
    print('Running preprocessing...')
    dataset = get_dataset('data/a1_RestaurantReviews_HistoricDump.tsv')
    no_stopwords = remove_stopwords(dataset)
    X, cv = preprocess(no_stopwords)
    y = dataset.iloc[:, -1].values
    np.save('data/X.npy', X)
    np.save('data/y.npy', y)
    save_preprocessing(cv)


def main():
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
