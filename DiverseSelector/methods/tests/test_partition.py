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

"""Test Partition-Based Selection Methods."""

import numpy as np
from DiverseSelector.methods.partition import DirectedSphereExclusion, GridPartitioning, Medoid
from DiverseSelector.methods.tests.common import generate_synthetic_data
from numpy.testing import assert_equal, assert_raises
import pytest


def test_directed_sphere_size_error():
    """Test DirectedSphereExclusion error when too many points requested."""
    x = np.array([[1, 9]] * 100)
    selector = DirectedSphereExclusion()
    assert_raises(ValueError, selector.select, x, size=105)


def test_directed_sphere_same_number_of_pts():
    """Test DirectSphereExclusion with `size` = number of points in dataset."""
    # (0,0) as the reference point
    x = np.array([[0,0],[0,1],[0,2],[0,3]])
    selector = DirectedSphereExclusion(r0=1, tol=0)
    selected = selector.select(arr=x, size=3)
    expected = [1,2,3]
    assert_equal(selected, expected)
    assert_equal(selector.r, 0.5)


def test_directed_sphere_exclusion_select_more_number_of_pts():
    """Test DirectSphereExclusion on points on the line with `size` < number of points in dataset."""
    # (0,0) as the reference point
    x = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]])
    selector = DirectedSphereExclusion(r0=0.5, tol=0)
    selected = selector.select(arr=x, size=3)
    expected = [1, 3, 5]
    assert_equal(selected, expected)
    assert_equal(selector.r, 1.0)


def test_directed_sphere_exclusion_on_line_with_():
    """Test Direct Sphere Exclusion on points on line with smaller distribution than the radius."""
    # (0,0) as the reference point
    x = np.array([[0, 0], [0, 1], [0, 1.1], [0, 1.2], [0, 2],
                  [0, 3], [0, 3.1], [0, 3.2], [0, 4], [0, 5], [0, 6]])
    selector = DirectedSphereExclusion(r0=0.5, tol=0)
    selected = selector.select(arr=x, size=3)
    expected = [1, 5, 9]
    assert_equal(selected, expected)
    assert_equal(selector.r, 1.0)


def test_directed_sphere_on_line_with_larger_radius():
    """Test Direct Sphere Exclusion on points on the line with a too large radius size."""
    # (0,0) as the reference point
    x = np.array([[0, 0], [0, 1], [0, 1.1], [0, 1.2], [0, 2],
                  [0, 3], [0, 3.1], [0, 3.2], [0, 4], [0, 5]])
    selector = DirectedSphereExclusion(r0=2.0, tol=0)
    selected = selector.select(arr=x, size=3)
    expected = [1, 5, 9]
    assert_equal(selected, expected)
    assert_equal(selector.r, 1.0)


@pytest.mark.parametrize("numb_pts", [10, 20, 30])
def test_grid_partitioning_equisized_independent_on_simple_example(numb_pts):
    r"""Test equisized_independent grid partitioning on an example each bin has only one point."""
    # Construct feature array where each molecule is known to be in which bin
    # The grid is a uniform grid from 0 to 10 in X-axis and 0 to 11 on y-axis.
    x = np.linspace(0, 10, numb_pts)
    y = np.linspace(0, 11, numb_pts)
    X, Y = np.meshgrid(x, y)
    grid = np.array([1.0, 0.0]) + np.vstack([X.ravel(), Y.ravel()]).T
    # Make one bin have an extra point
    grid = np.vstack((grid, np.array([1.1, 0.0])))

    # Here the number of cells should be equal to the number of points in each dimension
    #  excluding the extra point, so that the answer is unique/known.
    selector = GridPartitioning(numb_bins_axis=numb_pts)
    # Sort the points so that they're comparable to the expected answer.
    selected_ids = np.sort(selector.select(grid, size=len(grid) - 1))
    expected = np.arange(len(grid) - 1)
    assert_equal(selected_ids, expected)


@pytest.mark.parametrize("numb_pts", [4])
def test_grid_partitioning_equifrequent_independent_on_simple_example(numb_pts):
    r"""Test equisized_independent grid partitioning on an example each bin has only one point."""
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
    selector = GridPartitioning(numb_bins_axis=4, grid_method="equifrequent_independent")
    # Sort the points so that they're comparable to the expected answer.
    selected_ids = np.sort(selector.select(grid, size=len(grid) - 1))
    expected = np.arange(len(grid) - 1)
    assert_equal(selected_ids, expected)


def test_grid_partitioning_equisized_dependent_on_simple_example():
    r"""Test equisized_dependent grid partitioning on example that is different from independent."""
    # Construct feature array where each molecule is known to be in which bin
    grid = np.array([
        [0.0, 0.0],  # Corresponds to bin (0, 0)
        [0.0, 4.0],  # Corresponds to bin (0, 3)
        [1.0, 1.0],  # Corresponds to bin (1, 0)
        [1.0, 0.9],  # (1, 0)
        [1.0, 2.0],  # (1, 3)
        [2.0, 0.0],  # (2, 0)
        [2.0, 4.0],  # (2, 3)
        [3.0, 0.0],  # (3, 0)
        [3.0, 4.0],  # (3, 3)
        [3.0, 3.9]   # (3, 3)
    ])

    # The number of bins makes it so that it approxiamtely be a single point in each bin
    selector = GridPartitioning(numb_bins_axis=4, grid_method="equisized_dependent")
            # Two bins have an extra point in them and so has more diversity than other bins
    #   then the two expected molecules should be in those bins.
    selected_ids = selector.select(grid, size=2)
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
    selector = GridPartitioning(numb_bins_axis=numb_pts, grid_method="equifrequent_dependent")
    # Sort the points so that they're comparable to the expected answer.
    selected_ids = np.sort(selector.select(grid, size=len(grid) - 1))
    expected = np.arange(len(grid) - 1)
    assert_equal(selected_ids, expected)


@pytest.mark.parametrize("numb_pts", [10, 20, 30])
def test_grid_paritioning_equisized_dependent_same_as_independent_on_uniform_grid(numb_pts):
    r"""Test grid partitioning is the same between the two equisized methods on uniform grid in three-dimensions."""
    x = np.linspace(0, 10, numb_pts)
    y = np.linspace(0, 11, numb_pts)
    X = np.meshgrid(x, y, y)
    grid = np.vstack(list(map(np.ravel, X))).T
    grid = np.array([1.0, 0.0, 0.0]) + grid
    # Make one bin have an extra point
    # grid = np.vstack((grid, np.array([1.1, 0.0, 0.0])))

    # Here the number of cells should be equal to the number of points in each dimension
    #  excluding the extra point, so that the answer is unique/known.
    selector = GridPartitioning(numb_bins_axis=numb_pts, grid_method="equifrequent_independent")
    # Sort the points so that they're comparable to the expected answer.
    selected_ids_indep = np.sort(selector.select(grid, size=len(grid) - 1))

    selector = GridPartitioning(numb_bins_axis=numb_pts, grid_method="equifrequent_dependent")
    selected_ids_dep = np.sort(selector.select(grid, size=len(grid) - 1))
    assert_equal(selected_ids_dep, selected_ids_indep)


def test_raises_grid_partitioning():
    r"""Test raises error for grid partitioning."""
    grid = np.random.uniform(0.0, 1.0, size=(10, 3))

    assert_raises(TypeError, GridPartitioning, 5.0)        # Test number of axis should be integer
    assert_raises(TypeError, GridPartitioning, 5, 5.0)  # Test grid method should be string
    assert_raises(TypeError, GridPartitioning, 5, "string", [])  # Test random seed should be integer

    # Test the selector grid method is not the correct string
    selector = GridPartitioning(numb_bins_axis=5, grid_method="string")
    assert_raises(ValueError, selector.select_from_cluster, grid, 5)

    selector = GridPartitioning(numb_bins_axis=5)
    assert_raises(TypeError, selector.select_from_cluster, [5.0], 5)  # Test X is numpy array
    assert_raises(TypeError, selector.select_from_cluster, grid, 5.0)  # Test number selected should be int
    assert_raises(TypeError, selector.select_from_cluster, grid, 5, [5.0])

def test_medoid():
    """Testing Medoid class."""
    coords, _, _ = generate_synthetic_data(n_samples=100,
                                           n_features=2,
                                           n_clusters=1,
                                           pairwise_dist=True,
                                           metric="euclidean",
                                           random_state=42)

    coords_cluster, class_labels_cluster, _ = generate_synthetic_data(n_samples=100,
                                                                      n_features=2,
                                                                      n_clusters=3,
                                                                      pairwise_dist=True,
                                                                      metric="euclidean",
                                                                      random_state=42)
    selector = Medoid()
    selected_ids = selector.select(arr=coords_cluster, size=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [2, 73, 94, 86, 1, 50, 93, 78, 0, 54, 33, 72])

    selector = Medoid()
    selected_ids = selector.select(arr=coords, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [0, 95, 57, 41, 25, 9, 8, 6, 66, 1, 42, 82])
