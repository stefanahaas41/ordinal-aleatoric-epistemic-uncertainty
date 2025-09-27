import numpy as np

def variance_for_probabilities_binary(p):
    return p[0]*(1-p[0])

def mean_for_probabilities(probabilities):
    values = np.arange(len(probabilities))
    expected_value = probabilities @ values
    return expected_value

def variance_for_probabilities(probabilities):
    """
    Calculate Variance of Discrete Random Variable

    :param probabilities:
    :return:
    """
    values = np.arange(len(probabilities))
    expected_value = probabilities @ values
    squared_difference = (expected_value - values) ** 2
    return squared_difference @ probabilities


def standard_deviation_for_probabilities(probabilities):
    return np.sqrt(variance_for_probabilities(probabilities))