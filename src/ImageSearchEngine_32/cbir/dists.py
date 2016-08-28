import numpy as np


def chi2_distance(hist_a, hist_b, eps=1e-10):
    d = 0.5 * np.sum(((hist_a - hist_b) ** 2) / (hist_a + hist_b + eps))
    return d
