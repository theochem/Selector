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


# def test_gridpartitioning():
#     """Testing DirectedSphereExclusion class."""
#     coords, _, _ = generate_synthetic_data(n_samples=100,
#                                            n_features=2,
#                                            n_clusters=1,
#                                            pairwise_dist=True,
#                                            metric="euclidean",
#                                            random_state=42)

#     coords_cluster, class_labels_cluster, _ = generate_synthetic_data(n_samples=100,
#                                                                       n_features=2,
#                                                                       n_clusters=3,
#                                                                       pairwise_dist=True,
#                                                                       metric="euclidean",
#                                                                       random_state=42)
#     selector = GridPartitioning(cells=3)
#     selected_ids = selector.select(arr=coords_cluster, size=12, labels=class_labels_cluster)
#     # make sure all the selected indices are the same with expectation
#     assert_equal(selected_ids, [2, 25, 84, 56, 8, 70, 58, 78, 4, 46, 65, 29])

#     selector = GridPartitioning(cells=3)
#     selected_ids = selector.select(arr=coords, size=12)
#     # make sure all the selected indices are the same with expectation
#     assert_equal(selected_ids, [7, 55, 70, 57, 29, 91, 9, 65, 28, 11, 54, 88])


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
