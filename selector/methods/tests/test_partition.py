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
"""Test Partition-Based Selection Methods."""

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises

from selector.methods.partition import GridPartition, Medoid
from selector.methods.tests.common import generate_synthetic_data


@pytest.mark.parametrize("numb_pts", [4])
@pytest.mark.parametrize("method", ["equifrequent", "equisized"])
def test_grid_partitioning_independent_on_simple_example(numb_pts, method):
    r"""Test grid partitioning on an example each bin has only one point."""
    # Construct feature array where each molecule is known to be in which bin
    # The grid is a uniform grid from 0 to 10 in X-axis and 0 to 11 on y-axis.
    x = np.linspace(0, 3, numb_pts)
    y = np.linspace(0, 3, numb_pts)
    X, Y = np.meshgrid(x, y)
    grid = np.array([1.0, 0.0]) + np.vstack([X.ravel(), Y.ravel()]).T
    # Make one bin have an extra point
    grid = np.vstack((grid, np.array([1.1, 0.0])))

    # Here the number of cells should be equal to the number of points in each dimension
    #  excluding the extra point, so that the answer is unique/known.
    collector = GridPartition(nbins_axis=4, bin_method=f"{method}_independent")
    # Sort the points so that they're comparable to the expected answer.
    selected_ids = np.sort(collector.select(grid, size=len(grid) - 1))
    expected = np.arange(len(grid) - 1)
    assert_equal(selected_ids, expected)


def test_grid_partitioning_equisized_dependent_on_simple_example():
    r"""Test equisized_dependent grid partitioning on example that is different from independent."""
    # Construct feature array where each molecule is known to be in which bin
    grid = np.array(
        [
            [0.0, 0.0],  # Corresponds to bin (0, 0)
            [0.0, 4.0],  # Corresponds to bin (0, 3)
            [1.0, 1.0],  # Corresponds to bin (1, 0)
            [1.0, 0.9],  # Corresponds to bin (1, 0)
            [1.0, 2.0],  # Corresponds to bin (1, 3)
            [2.0, 0.0],  # Corresponds to bin (2, 0)
            [2.0, 4.0],  # Corresponds to bin (2, 3)
            [3.0, 0.0],  # Corresponds to bin (3, 0)
            [3.0, 4.0],  # Corresponds to bin (3, 3)
            [3.0, 3.9],  # Corresponds to bin (3, 3)
        ]
    )

    # The number of bins makes it so that it approximately be a single point in each bin
    collector = GridPartition(nbins_axis=4, bin_method="equisized_dependent")
    # Two bins have an extra point in them and so has more diversity than other bins
    #   then the two expected molecules should be in those bins.
    selected_ids = collector.select(grid, size=2, labels=None)
    right_molecules = True
    if not (2 in selected_ids or 3 in selected_ids):
        right_molecules = False
    if not (8 in selected_ids or 9 in selected_ids):
        right_molecules = False
    assert right_molecules, "The correct points were selected"


@pytest.mark.parametrize("numb_pts", [4])
def test_grid_partitioning_equifrequent_dependent_on_simple_example(numb_pts):
    r"""Test equifrequent dependent grid partitioning on an example where each bin has only one point."""
    # Construct feature array where each molecule is known to be in which bin
    # The grid is a uniform grid from 0 to 10 in X-axis and 0 to 11 on y-axis.
    x = np.linspace(0, 3, numb_pts)
    y = np.linspace(0, 3, numb_pts)
    X, Y = np.meshgrid(x, y)
    grid = np.array([1.0, 0.0]) + np.vstack([X.ravel(), Y.ravel()]).T
    # Make one bin have an extra point
    grid = np.vstack((grid, np.array([1.1, 0.0])))

    # Here the number of cells should be equal to the number of points in each dimension
    #  excluding the extra point, so that the answer is unique/known.
    collector = GridPartition(nbins_axis=numb_pts, bin_method="equifrequent_dependent")
    # Sort the points so that they're comparable to the expected answer.
    selected_ids = np.sort(collector.select(grid, size=len(grid) - 1))
    expected = np.arange(len(grid) - 1)
    assert_equal(selected_ids, expected)


@pytest.mark.parametrize("numb_pts", [10, 20, 30])
@pytest.mark.parametrize("method", ["equifrequent", "equisized"])
def test_bins_from_both_methods_dependent_same_as_independent_on_uniform_grid(numb_pts, method):
    r"""Test bins is the same between the two equisized methods on uniform grid in three-dimensions."""
    x = np.linspace(0, 10, numb_pts)
    y = np.linspace(0, 11, numb_pts)
    X = np.meshgrid(x, y, y)
    grid = np.vstack(list(map(np.ravel, X))).T
    grid = np.array([1.0, 0.0, 0.0]) + grid

    # Here the number of cells should be equal to the number of points in each dimension
    #  excluding the extra point, so that the answer is unique/known.
    collector_indept = GridPartition(nbins_axis=numb_pts, bin_method=f"{method}_independent")
    collector_depend = GridPartition(nbins_axis=numb_pts, bin_method=f"{method}_dependent")

    # Get the bins from the method
    bins_indept = collector_indept.get_bins_from_method(grid)
    bins_dept = collector_depend.get_bins_from_method(grid)

    # Test the bins are the same
    for key in bins_indept.keys():
        assert_equal(bins_dept[key], bins_indept[key])


def test_raises_grid_partitioning():
    r"""Test raises error for grid partitioning."""
    grid = np.random.uniform(0.0, 1.0, size=(10, 3))

    assert_raises(TypeError, GridPartition, 5.0)  # Test number of axis should be integer
    assert_raises(TypeError, GridPartition, 5, 5.0)  # Test grid method should be string
    assert_raises(TypeError, GridPartition, 5, "string", [])  # Test random seed should be integer

    # Test the collector grid method is not the correct string
    collector = GridPartition(nbins_axis=5, bin_method="string")
    assert_raises(ValueError, collector.select_from_cluster, grid, 5)

    collector = GridPartition(nbins_axis=5)
    assert_raises(TypeError, collector.select_from_cluster, [5.0], 5)  # Test X is numpy array
    assert_raises(
        TypeError, collector.select_from_cluster, grid, 5.0
    )  # Test number selected should be int
    assert_raises(TypeError, collector.select_from_cluster, grid, 5, [5.0])


def test_medoid():
    """Testing Medoid class."""
    coords, _, _ = generate_synthetic_data(
        n_samples=100,
        n_features=2,
        n_clusters=1,
        pairwise_dist=True,
        metric="euclidean",
        random_state=42,
    )

    coords_cluster, class_labels_cluster, _ = generate_synthetic_data(
        n_samples=100,
        n_features=2,
        n_clusters=3,
        pairwise_dist=True,
        metric="euclidean",
        random_state=42,
    )
    collector = Medoid()
    selected_ids = collector.select(coords_cluster, size=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [2, 73, 94, 86, 1, 50, 93, 78, 0, 54, 33, 72])

    collector = Medoid()
    selected_ids = collector.select(coords, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [0, 95, 57, 41, 25, 9, 8, 6, 66, 1, 42, 82])

    # test the case where KD-Tree query return is an integer
    features = np.array([[1.5, 2.8], [2.3, 3.8], [1.5, 2.8], [4.0, 5.9]])
    selector = Medoid()
    selected_ids = selector.select(features, size=2)
    assert_equal(selected_ids, [0, 3])
