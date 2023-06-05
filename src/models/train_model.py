"""Module to train a model and save it to a specified path"""

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def train_model(X, y, random_state=0):
    """
    train model and return the trained model
    args:
        X: numpy array
        y: numpy array
    """
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print(confusion_mat)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    metrics = {"accuracy": accuracy}
    with open("ml_models/metrics/metrics.json", "w") as outfile:
        json.dump(metrics, outfile)
    return classifier


def load_training_data(folder_path):
    """
    load preprocessed training data from a specified folder path
    """
    X = np.load(f'{folder_path}/X.npy')
    y = np.load(f'{folder_path}/y.npy')
    return X, y


def save_model(classifier, path):
    """
    save model to a specified path
    args:
        classifier: sklearn classifier
        path: str
    """
    joblib.dump(classifier, path)


def main():
    """Main function to run script"""
    print("loading_data")
    X, y = load_training_data('data/processed')
    print("training_model")
    classifier = train_model(X, y)
    print("saving_model")
    save_model(classifier, "ml_models/c2_Classifier_Sentiment_Model")


if __name__ == "__main__":
    main()
