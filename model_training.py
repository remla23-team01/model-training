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
    dataset = pd.read_csv(filename, delimiter = '\t', quoting = 3)
    return dataset


def remove_stopwords(dataset):
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus=[]

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
        
    return corpus


def preprocess(text):
    cv = CountVectorizer(max_features = 1420)
    X = cv.fit_transform(text).toarray()
    return X, cv
    

def save_preproccessing(cv):
    bow_path = 'c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'c2_Classifier_Sentiment_Model') 

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


def main():
    dataset = get_dataset('a1_RestaurantReviews_HistoricDump.tsv')
    no_stopwords = remove_stopwords(dataset)
    X, cv = preprocess(no_stopwords)
    save_preproccessing(cv)
    y = dataset.iloc[:, -1].values
    train_model(X, y)


if __name__ == '__main__':
    main()