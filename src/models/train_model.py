import joblib
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

def train_model(X, y):
    """
    train model and return the trained model
    args:
        X: numpy array
        y: numpy array
    """
    print('Training model...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    
    metrics = {'accuracy': accuracy}
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
    print("loading_data")
    X, y = load_training_data('data/processed')
    
    print("training_model")
    classifier = train_model(X, y)
    
    print("saving_model")
    save_model(classifier, 'ml_models/c2_Classifier_Sentiment_Model')

if __name__ == '__main__':
    main()