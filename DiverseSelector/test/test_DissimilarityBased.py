import sys
from ..dissimilarity_based import DissimilaritySelection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from numpy.testing import assert_equal



def test_MinMaxSelector():
    """Testing the MinMax selection algorithm"""
    data, labels = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)
    arr_dist = pairwise_distances(data)
    model = DissimilaritySelection(num_selected=10, arr_dist=arr_dist)
    model.starting_idx = 0
    selected = model.select()
    assert_equal([0, 94, 3, 50, 64, 85, 93, 83, 34], selected)
