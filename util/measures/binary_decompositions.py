import unittest

import numpy as np
from scipy.stats import entropy


def binary_one_vs_rest_entropy(p):
    if isinstance(p, list):
        p = np.array(p)
    sum = 0
    for i in range(len(p)):
        m = np.ma.array(p, mask=False)
        m.mask[i] = True
        bin_prob = np.array([p[i], m.sum()])
        sum += entropy(bin_prob, base=2)
    return sum

def binary_one_vs_rest_variance(p):
    if isinstance(p, list):
        p = np.array(p)
    sum = 0
    for i in range(len(p)):
        m = np.ma.array(p, mask=False)
        m.mask[i] = True
        m = p[i] * m.sum()
        sum += m
    return sum

def binary_ordinal_entropy(p):
    if isinstance(p, list):
        p = np.array(p)
    sum = 0
    for i in range(len(p) - 1):
        bin_prob = np.array([np.sum(p[:i + 1]), np.sum(p[i + 1:])])
        sum += entropy(bin_prob, base=2)
    return sum


def eu_binary_variance_one_vs_rest(p):
    rows = p.shape[0]
    cols = p.shape[1]
    mean_probs = []
    for i in range(cols):
        mean_probs.append(np.sum(p[:, i]) / rows)
    variance = np.zeros((rows, cols))
    for i in range(0, cols):
        variance[:, i] = (p[:, i] - mean_probs[i]) ** 2
    eu = np.sum(np.mean(variance, axis=0))
    return eu

def eu_binary_variance( p):
    rows = p.shape[0]
    cols = p.shape[1]
    mean_expected_values = []
    cumulative_probs = np.zeros((rows, cols - 1))
    for i in range(1, cols):
        mean_expected_values.append(np.sum(p[:, :i]) / rows)
        cumulative_probs[:, i - 1] = np.sum(p[:, :i], axis=1)
    variance = np.zeros((rows, cols - 1))
    for i in range(0, cols - 1):
        variance[:, i] = (cumulative_probs[:, i] - mean_expected_values[i]) ** 2
    eu = np.sum(np.mean(variance, axis=0))
    return eu

def binary_ordinal_variance(p):
    if isinstance(p, list):
        p = np.array(p)
    sum = 0
    for i in range(len(p) - 1):
        m = np.sum(p[:i + 1]) * np.sum(p[i + 1:])
        # print("Variance ", m)
        sum += m
    # print("Variance ", sum)
    #norm = (len(p) - 1) / 4
    return sum #/ norm


def binary_ordinal_margin(p):
    if isinstance(p, list):
        p = np.array(p)
    sum = 0
    for i in range(len(p) - 1):
        bin_prob = np.array([np.sum(p[:i + 1]), np.sum(p[i + 1:])])
        diff = abs(bin_prob[0] - bin_prob[1])
        sum += 1 - diff
    return sum


def binary_ordinal_margin_norm(p):
    if isinstance(p, list):
        p = np.array(p)
    sum = 0
    for i in range(len(p)):
        diff = abs(np.sum(p[:i + 1]) - np.sum(p[i + 1:]))
        # bin_prob = np.array([np.sum(p[:i + 1]), np.sum(p[i + 1:])])
        # diff = abs(bin_prob[0] - bin_prob[1])
        print("Bin (Margin) ", diff)
        sum += diff
    norm = (len(p) - 1)
    return len(p) / norm - sum / norm


class BinMarginTests(unittest.TestCase):
    def test_bimodal(self):
        p = [0.5, 0, 0.5]
        binary_ord_conf = binary_ordinal_margin(p)
        self.assertEqual(2, binary_ord_conf)

    def test_uniform(self):
        p = [1 / 3, 1 / 3, 1 / 3]
        binary_ord_conf = binary_ordinal_margin(p)
        self.assertEqual(1.3333333333333335, binary_ord_conf)

    def test_dirac(self):
        p = [0, 1, 0]
        binary_ord_conf = binary_ordinal_margin(p)
        self.assertEqual(0, binary_ord_conf)


class BinVarianceTests(unittest.TestCase):

    def test_bimodal_bin(self):
        p = [0.5, 0.5]
        binary_ord_variance = binary_ordinal_variance(p)
        self.assertEqual(0.25, binary_ord_variance)

    def test_bimodal(self):
        p = [0.5, 0, 0.5]
        binary_ord_variance = binary_ordinal_variance(p)
        self.assertEqual(0.5, binary_ord_variance)

    def test_uniform(self):
        p = [1 / 3, 1 / 3, 1 / 3]
        binary_ord_variance = binary_ordinal_variance(p)
        self.assertEqual(0.4444444444444444, binary_ord_variance)

    def test_dirac(self):
        p = [0, 1, 0]
        binary_ord_variance = binary_ordinal_variance(p)
        self.assertEqual(0, binary_ord_variance)


class BinEntropyTests(unittest.TestCase):
    def test_bimodal(self):
        p = [0.5, 0, 0.5]
        bin_orinal_entropy = binary_ordinal_entropy(p)
        self.assertEqual(1.3862943611198906, bin_orinal_entropy)

    def test_uniform(self):
        p = [1 / 3, 1 / 3, 1 / 3]
        bin_orinal_entropy = binary_ordinal_entropy(p)
        self.assertEqual(1.2730283365896256, bin_orinal_entropy)

    def test_dirac(self):
        p = [0, 1, 0]
        bin_orinal_entropy = binary_ordinal_entropy(p)
        self.assertEqual(0, bin_orinal_entropy)
