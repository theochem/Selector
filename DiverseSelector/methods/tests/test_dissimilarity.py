# The DiverseSelector library provides a set of tools to select molecule
# subset with maximum molecular diversity.
#
# Copyright (C) 2022 The QC-Devs Community
#
# This file is part of DiverseSelector.
#
# DiverseSelector is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# DiverseSelector is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --

"""Test Dissimilarity-Based Selection Methods."""


from DiverseSelector.methods.dissimilarity import MaxMin, MaxSum, OptiSim
import numpy as np
from numpy.testing import assert_equal
from sklearn.metrics import pairwise_distances
from DiverseSelector.methods.tests.common import generate_synthetic_data


def test_maxmin():
    """Testing the MaxMin class."""
    # generate random data points belonging to one cluster - pairwise distance matrix
    _, _, arr_dist = generate_synthetic_data(n_samples=100,
                                             n_features=2,
                                             n_clusters=1,
                                             pairwise_dist=True,
                                             metric="euclidean",
                                             random_state=42)

    # generate random data points belonging to multiple clusters - class labels and pairwise distance matrix
    _, class_labels_cluster, arr_dist_cluster = generate_synthetic_data(n_samples=100,
                                                                        n_features=2,
                                                                        n_clusters=3,
                                                                        pairwise_dist=True,
                                                                        metric="euclidean",
                                                                        random_state=42)

    # use MaxMin algorithm to select points from clustered data
    selector = MaxMin()
    selected_ids = selector.select(arr=arr_dist_cluster,
                                   size=12,
                                   labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [41, 34, 94, 85, 51, 50, 66, 78, 21, 64, 29, 83])

    # use MaxMin algorithm to select points from non-clustered data
    selector = MaxMin()
    selected_ids = selector.select(arr=arr_dist, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [85, 57, 41, 25, 9, 62, 29, 65, 81, 61, 60, 97])

    # use MaxMin algorithm, this time instantiating with a distance metric
    selector = MaxMin(lambda x: pairwise_distances(x, metric='euclidean'))
    simple_coords = np.array([[0, 0],
                              [2, 0],
                              [0, 2],
                              [2, 2],
                              [-10, -10]])
    # provide coordinates rather than pairwise distance matrix to selector
    selected_ids = selector.select(arr=simple_coords, num_selected=3)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [0, 4, 3])

    # generating mocked clusters
    np.random.seed(42)
    cluster_one = np.random.normal(0, 1, (3, 2))
    cluster_two = np.random.normal(10, 1, (6, 2))
    cluster_three = np.random.normal(20, 1, (10, 2))
    labels_mocked = np.hstack([[0 for i in range(3)],
                               [1 for i in range(6)],
                               [2 for i in range(10)]])
    mocked_cluster_coords = np.vstack([cluster_one, cluster_two, cluster_three])

    # selecting molecules
    selector = MaxMin(lambda x: pairwise_distances(x, metric='euclidean'))
    selected_mocked = selector.select(mocked_cluster_coords,
                                      size=15,
                                      labels=labels_mocked)
    assert_equal(selected_mocked, [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 15, 10, 13, 9, 18])


def test_maxsum():
    """Testing MaxSum class."""
    # generate random data points belonging to one cluster - coordinates
    coords, _, _ = generate_synthetic_data(n_samples=100,
                                           n_features=2,
                                           n_clusters=1,
                                           pairwise_dist=True,
                                           metric="euclidean",
                                           random_state=42)

    # generate random data points belonging to multiple clusters - coordinates and class labels
    coords_cluster, class_labels_cluster, _ = generate_synthetic_data(n_samples=100,
                                                                      n_features=2,
                                                                      n_clusters=3,
                                                                      pairwise_dist=True,
                                                                      metric="euclidean",
                                                                      random_state=42)

    # use MaxSum algorithm to select points from clustered data, instantiating with euclidean distance metric
    selector = MaxSum(lambda x: pairwise_distances(x, metric='euclidean'))
    selected_ids = selector.select(arr=coords_cluster, size=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [41, 34, 85, 94, 51, 50, 78, 66, 21, 64, 0, 83])

    # use MaxSum algorithm to select points from non-clustered data, instantiating with euclidean distance metric
    selector = MaxSum(lambda x: pairwise_distances(x, metric='euclidean'))
    selected_ids = selector.select(arr=coords, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [85, 57, 25, 41, 95, 9, 21, 8, 13, 68, 37, 54])


def test_optisim():
    """Testing OptiSim class."""
    # generate random data points belonging to one cluster - coordinates and pairwise distance matrix
    coords, _, arr_dist = generate_synthetic_data(n_samples=100,
                                                  n_features=2,
                                                  n_clusters=1,
                                                  pairwise_dist=True,
                                                  metric="euclidean",
                                                  random_state=42)

    # generate random data points belonging to multiple clusters - coordinates and class labels
    coords_cluster, class_labels_cluster, _ = generate_synthetic_data(n_samples=100,
                                                                      n_features=2,
                                                                      n_clusters=3,
                                                                      pairwise_dist=True,
                                                                      metric="euclidean",
                                                                      random_state=42)

    # use OptiSim algorithm to select points from clustered data
    selector = OptiSim()
    selected_ids = selector.select(arr=coords_cluster, size=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [2, 85, 86, 59, 1, 66, 50, 68, 0, 64, 83, 72])

    # use OptiSim algorithm to select points from non-clustered data
    selector = OptiSim()
    selected_ids = selector.select(arr=coords, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [0, 8, 55, 37, 41, 13, 12, 42, 6, 30, 57, 76])

    # tester to check if OptiSim gives same results as MaxMin for k=>infinity
    selector = OptiSim(start_id=85, k=999999)
    selected_ids_optisim = selector.select(arr=coords, size=12)
    selector = MaxMin()
    selected_ids_maxmin = selector.select(arr=arr_dist, size=12)
    assert_equal(selected_ids_optisim, selected_ids_maxmin)
