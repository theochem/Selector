# -*- coding: utf-8 -*-
#
# The Selector is a Python library of algorithms for selecting diverse
# subsets of data for machine-learning.
#
# Copyright (C) 2022-2024 The QC-Devs Community
#
# This file is part of Selector.
#
# Selector is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Selector is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Test selector/methods/distance.py."""

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises
from sklearn.metrics import pairwise_distances

from selector.methods.distance import DISE, MaxMin, MaxSum, OptiSim
from selector.methods.tests.common import generate_synthetic_data


def test_maxmin():
    """Testing the MaxMin class."""
    # generate random data points belonging to one cluster - pairwise distance matrix
    _, _, arr_dist = generate_synthetic_data(
        n_samples=100,
        n_features=2,
        n_clusters=1,
        pairwise_dist=True,
        metric="euclidean",
        random_state=42,
    )

    # generate random data points belonging to multiple clusters - class labels and pairwise distance matrix
    _, class_labels_cluster, arr_dist_cluster = generate_synthetic_data(
        n_samples=100,
        n_features=2,
        n_clusters=3,
        pairwise_dist=True,
        metric="euclidean",
        random_state=42,
    )

    # use MaxMin algorithm to select points from clustered data
    collector = MaxMin()
    selected_ids = collector.select(arr_dist_cluster, size=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [41, 34, 94, 85, 51, 50, 66, 78, 21, 64, 29, 83])

    # use MaxMin algorithm to select points from non-clustered data
    collector = MaxMin()
    selected_ids = collector.select(arr_dist, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [85, 57, 41, 25, 9, 62, 29, 65, 81, 61, 60, 97])

    # use MaxMin algorithm to select points from non-clustered data with "medoid" as the reference point
    collector_medoid_1 = MaxMin(ref_index=None)
    selected_ids_medoid_1 = collector_medoid_1.select(arr_dist, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids_medoid_1, [85, 57, 41, 25, 9, 62, 29, 65, 81, 61, 60, 97])

    # use MaxMin algorithm to select points from non-clustered data with "None" for the reference point
    collector_medoid_2 = MaxMin(ref_index=None)
    selected_ids_medoid_2 = collector_medoid_2.select(arr_dist, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids_medoid_2, [85, 57, 41, 25, 9, 62, 29, 65, 81, 61, 60, 97])

    # use MaxMin algorithm to select points from non-clustered data with float as the reference point
    collector_float = MaxMin(ref_index=85)
    selected_ids_float = collector_float.select(arr_dist, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids_float, [85, 57, 41, 25, 9, 62, 29, 65, 81, 61, 60, 97])

    # use MaxMin algorithm to select points from non-clustered data with a predefined list as the reference point
    collector_float = MaxMin(ref_index=[85, 57, 41, 25])
    selected_ids_float = collector_float.select(arr_dist, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids_float, [85, 57, 41, 25, 9, 62, 29, 65, 81, 61, 60, 97])

    # test failing case when ref_index is not a valid index
    with pytest.raises(ValueError):
        collector_float = MaxMin(ref_index=-3)
        selected_ids_float = collector_float.select(arr_dist, size=12)
    # test failing case when ref_index contains a complex number
    with pytest.raises(ValueError):
        collector_float = MaxMin(ref_index=[1 + 5j, 2, 5])
        selected_ids_float = collector_float.select(arr_dist, size=12)
    # test failing case when ref_index contains a negative number
    with pytest.raises(ValueError):
        collector_float = MaxMin(ref_index=[-1, 2, 5])
        selected_ids_float = collector_float.select(arr_dist, size=12)

    # use MaxMin algorithm, this time instantiating with a distance metric
    collector = MaxMin(fun_dist=lambda x: pairwise_distances(x, metric="euclidean"))
    simple_coords = np.array([[0, 0], [2, 0], [0, 2], [2, 2], [-10, -10]])
    # provide coordinates rather than pairwise distance matrix to collector
    selected_ids = collector.select(x=simple_coords, size=3)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [0, 4, 3])

    # generating mocked clusters
    np.random.seed(42)
    cluster_one = np.random.normal(0, 1, (3, 2))
    cluster_two = np.random.normal(10, 1, (6, 2))
    cluster_three = np.random.normal(20, 1, (10, 2))
    labels_mocked = np.hstack(
        [[0 for i in range(3)], [1 for i in range(6)], [2 for i in range(10)]]
    )
    mocked_cluster_coords = np.vstack([cluster_one, cluster_two, cluster_three])

    # selecting molecules
    collector = MaxMin(lambda x: pairwise_distances(x, metric="euclidean"))
    selected_mocked = collector.select(mocked_cluster_coords, size=15, labels=labels_mocked)
    assert_equal(selected_mocked, [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 15, 10, 13, 9, 18])


def test_maxmin_invalid_input():
    """Testing MaxMin class with invalid input."""
    # case when the distance matrix is not square
    x_dist = np.array([[0, 1], [1, 0], [4, 9]])
    with pytest.raises(ValueError):
        collector = MaxMin(ref_index=0)
        _ = collector.select(x_dist, size=2)

    # case when the distance matrix is not symmetric
    x_dist = np.array([[0, 1, 2, 1], [1, 1, 0, 3], [4, 9, 4, 0], [6, 5, 6, 7]])
    with pytest.raises(ValueError):
        collector = MaxMin(ref_index=0)
        _ = collector.select(x_dist, size=2)


def test_maxsum_clustered_data():
    """Testing MaxSum class."""
    # generate random data points belonging to multiple clusters - coordinates and class labels
    coords_cluster, class_labels_cluster, coords_cluster_dist = generate_synthetic_data(
        n_samples=100,
        n_features=2,
        n_clusters=3,
        pairwise_dist=True,
        metric="euclidean",
        random_state=42,
    )

    # use MaxSum algorithm to select points from clustered data, instantiating with euclidean distance metric
    collector = MaxSum(fun_dist=lambda x: pairwise_distances(x, metric="euclidean"), ref_index=None)
    selected_ids = collector.select(coords_cluster, size=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [41, 34, 85, 94, 51, 50, 78, 66, 21, 64, 0, 83])

    # use MaxSum algorithm to select points from clustered data without instantiating with euclidean distance metric
    collector = MaxSum(ref_index=None)
    selected_ids = collector.select(coords_cluster_dist, size=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [41, 34, 85, 94, 51, 50, 78, 66, 21, 64, 0, 83])

    # check that ValueError is raised when number of points requested is greater than number of points in array
    with pytest.raises(ValueError):
        selected_ids = collector.select_from_cluster(
            coords_cluster, size=101, labels=class_labels_cluster
        )


def test_maxsum_non_clustered_data():
    """Testing MaxSum class with non-clustered data."""
    # generate random data points belonging to one cluster - coordinates
    coords, _, _ = generate_synthetic_data(
        n_samples=100,
        n_features=2,
        n_clusters=1,
        pairwise_dist=True,
        metric="euclidean",
        random_state=42,
    )
    # use MaxSum algorithm to select points from non-clustered data, instantiating with euclidean distance metric
    collector = MaxSum(fun_dist=lambda x: pairwise_distances(x, metric="euclidean"), ref_index=None)
    selected_ids = collector.select(coords, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [85, 57, 25, 41, 95, 9, 21, 8, 13, 68, 37, 54])

    # use MaxSum algorithm to select points from non-clustered data, instantiating with euclidean
    # distance metric and using "medoid" as the reference point
    collector = MaxSum(fun_dist=lambda x: pairwise_distances(x, metric="euclidean"), ref_index=None)
    selected_ids = collector.select(coords, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [85, 57, 25, 41, 95, 9, 21, 8, 13, 68, 37, 54])

    # use MaxSum algorithm to select points from non-clustered data, instantiating with euclidean
    # distance metric and using a list as the reference points
    collector = MaxSum(
        fun_dist=lambda x: pairwise_distances(x, metric="euclidean"),
        ref_index=[85, 57, 25, 41, 95],
    )
    selected_ids = collector.select(coords, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [85, 57, 25, 41, 95, 9, 21, 8, 13, 68, 37, 54])

    # use MaxSum algorithm to select points from non-clustered data, instantiating with euclidean
    # distance metric and using an invalid reference point
    with pytest.raises(ValueError):
        collector = MaxSum(
            fun_dist=lambda x: pairwise_distances(x, metric="euclidean"), ref_index=-1
        )
        selected_ids = collector.select(coords, size=12)


def test_maxsum_invalid_input():
    """Testing MaxSum class with invalid input."""
    # case when the distance matrix is not square
    x_dist = np.array([[0, 1], [1, 0], [4, 9]])
    with pytest.raises(ValueError):
        collector = MaxSum(ref_index=0)
        _ = collector.select(x_dist, size=2)

    # case when the distance matrix is not square
    x_dist = np.array([[0, 1, 2], [1, 0, 3], [4, 9, 0], [5, 6, 7]])
    with pytest.raises(ValueError):
        collector = MaxSum(ref_index=0)
        _ = collector.select(x_dist, size=2)

    # case when the distance matrix is not symmetric
    x_dist = np.array([[0, 1, 2, 1], [1, 1, 0, 3], [4, 9, 4, 0], [6, 5, 6, 7]])
    with pytest.raises(ValueError):
        collector = MaxSum(ref_index=0)
        _ = collector.select(x_dist, size=2)


def test_optisim():
    """Testing OptiSim class."""
    # generate random data points belonging to one cluster - coordinates and pairwise distance matrix
    coords, _, arr_dist = generate_synthetic_data(
        n_samples=100,
        n_features=2,
        n_clusters=1,
        pairwise_dist=True,
        metric="euclidean",
        random_state=42,
    )

    # generate random data points belonging to multiple clusters - coordinates and class labels
    coords_cluster, class_labels_cluster, _ = generate_synthetic_data(
        n_samples=100,
        n_features=2,
        n_clusters=3,
        pairwise_dist=True,
        metric="euclidean",
        random_state=42,
    )

    # use OptiSim algorithm to select points from clustered data
    collector = OptiSim(ref_index=0)
    selected_ids = collector.select(coords_cluster, size=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    # assert_equal(selected_ids, [2, 85, 86, 59, 1, 66, 50, 68, 0, 64, 83, 72])

    # use OptiSim algorithm to select points from non-clustered data
    collector = OptiSim(ref_index=0)
    selected_ids = collector.select(coords, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [0, 8, 55, 37, 41, 13, 12, 42, 6, 30, 57, 76])

    # check if OptiSim gives same results as MaxMin for k=>infinity
    collector = OptiSim(ref_index=85, k=999999)
    selected_ids_optisim = collector.select(coords, size=12)
    collector = MaxMin()
    selected_ids_maxmin = collector.select(arr_dist, size=12)
    assert_equal(selected_ids_optisim, selected_ids_maxmin)

    # test with invalid ref_index
    with pytest.raises(ValueError):
        collector = OptiSim(ref_index=10000)
        _ = collector.select(coords, size=12)


def test_directed_sphere_size_error():
    """Test DirectedSphereExclusion error when too many points requested."""
    x = np.array([[1, 9]] * 100)
    collector = DISE()
    assert_raises(ValueError, collector.select, x, size=105)


def test_directed_sphere_same_number_of_pts():
    """Test DirectSphereExclusion with `size` = number of points in dataset."""
    # (0,0) as the reference point
    x = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    collector = DISE(r0=1, tol=0, ref_index=0)
    selected = collector.select(x, size=2)
    assert_equal(selected, [0, 2])
    assert_equal(collector.r, 1)


def test_directed_sphere_same_number_of_pts_None():
    """Test DirectSphereExclusion with `size` = number of points in dataset with the ref_index None."""
    # None as the reference point
    x = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]])
    collector = DISE(r0=1, tol=0, ref_index=None)
    selected = collector.select(x, size=3)
    assert_equal(selected, [2, 0, 4])
    assert_equal(collector.r, 1)


def test_directed_sphere_exclusion_select_more_number_of_pts():
    """Test DirectSphereExclusion on points on the line with `size` < number of points in dataset."""
    # (0,0) as the reference point
    x = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]])
    collector = DISE(r0=0.5, tol=0, ref_index=0)
    selected = collector.select(x, size=3)
    expected = [0, 3, 6]
    assert_equal(selected, expected)
    assert_equal(collector.r, 2.0)


def test_directed_sphere_exclusion_on_line_with_smaller_radius():
    """Test Direct Sphere Exclusion on points on line with smaller distribution than the radius."""
    # (0,0) as the reference point
    x = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 1.1],
            [0, 1.2],
            [0, 2],
            [0, 3],
            [0, 3.1],
            [0, 3.2],
            [0, 4],
            [0, 5],
            [0, 6],
        ]
    )
    collector = DISE(r0=0.5, tol=1, ref_index=1)
    selected = collector.select(x, size=3)
    expected = [1, 5, 9]
    assert_equal(selected, expected)
    assert_equal(collector.r, 1.0)


def test_directed_sphere_on_line_with_larger_radius():
    """Test Direct Sphere Exclusion on points on the line with a too large radius size."""
    # (0,0) as the reference point
    x = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 1.1],
            [0, 1.2],
            [0, 2],
            [0, 3],
            [0, 3.1],
            [0, 3.2],
            [0, 4],
            [0, 5],
        ]
    )
    collector = DISE(r0=2.0, tol=0, p=2.0, ref_index=1)
    selected = collector.select(x, size=3)
    expected = [1, 5, 9]
    assert_equal(selected, expected)
    assert_equal(collector.r, 1.0)
