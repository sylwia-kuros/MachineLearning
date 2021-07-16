from typing import List

import numpy as np


def email_features(word_indices: List[int]) -> np.ndarray:
    """Convert a list of word IDs into a feature vector.

    :param word_indices: a list of word IDs
    :return: a feature vector from the word indices (a row vector)
    """

    # Total number of words in the dictionary
    n_words = 1899

    # Feature vector initialization
    feature_vector = np.zeros(n_words)

    for index in word_indices:
        feature_vector[index - 1] = 1

    return feature_vector.reshape(1, -1)
