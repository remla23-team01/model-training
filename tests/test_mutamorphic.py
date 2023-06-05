import joblib
import numpy as np
import pandas as pd
import pytest
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split

from model_training import get_dataset, preprocess, remove_stopwords
from src.train_model import train_model


class TestMutamorphic:
    def generate_mutation_for_review(self, review):
        """
        Generates a mutation for a review by replacing a synonym
        to a random word.
        """

        words = review.split()
        random_index = np.random.randint(0, len(words))
        word_to_mutate = words[random_index]

        word_synonyms = []
        for synonyms in wordnet.synsets(word_to_mutate):
            for lemma in synonyms.lemmas():
                word_synonyms.append(lemma.name())

        if len(word_synonyms) == 0:
            return review

        random_synonym_index = np.random.randint(0, len(word_synonyms))
        mutated_review = review.replace(
            word_to_mutate, word_synonyms[random_synonym_index]
        )

        return mutated_review

    def test_mutamorphic(self):
        """
        Test mutaphormic.
        """
        model = joblib.load("ml_models/c2_Classifier_Sentiment_Model")

        dataset = get_dataset("data/a1_RestaurantReviews_HistoricDump.tsv")

        dataset_preprocessed = remove_stopwords(dataset)

        _, X_test, _, y_test = train_test_split(
            dataset_preprocessed,
            dataset["Liked"],
            test_size=0.20,
            random_state=0,
        )

        # mutate X_test
        X_test_mutated = X_test.apply(self.generate_mutation_for_review)

        # preprocess
        X_test, _ = preprocess(X_test)
        X_test_mutated, _ = preprocess(X_test_mutated)

        # test accuracy
        accuracy = sum(model.predict(X_test) == y_test) / len(y_test)
        mutated_accuracy = sum(model.predict(X_test_mutated) == y_test) / len(
            y_test
        )

        assert abs(accuracy - mutated_accuracy) < 0.2
