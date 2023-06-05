import numpy as np
import pytest
from nltk.corpus import wordnet


@pytest.fixture
def X():
    """
    Load X data
    """
    return np.load("data/X.npy")


@pytest.fixture
def y():
    """
    Load y data
    """
    return np.load("data/y.npy")


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

    def test_mutamorphic(self, X, y):
        print("I like it")
        print(self.generate_mutation_for_review("I like it"))
        assert True
