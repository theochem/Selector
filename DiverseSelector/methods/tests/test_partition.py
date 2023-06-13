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

from DiverseSelector.methods.partition import DirectedSphereExclusion, GridPartitioning, Medoid
from DiverseSelector.methods.tests.common import generate_synthetic_data
from numpy.testing import assert_equal



def test_directedsphereexclusion():
    """Testing DirectedSphereExclusion class."""
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
    selector = DirectedSphereExclusion()
    selected_ids = selector.select(arr=coords_cluster, size=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [95, 14, 88, 84, 76, 68, 93, 50, 29, 19, 54])

    selector = DirectedSphereExclusion()
    selected_ids = selector.select(arr=coords, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [17, 92, 64, 6, 12, 76, 10, 87, 73, 66, 11, 57])


def test_gridpartitioning():
    """Testing DirectedSphereExclusion class."""
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
    selector = GridPartitioning(cells=3)
    selected_ids = selector.select(arr=coords_cluster, size=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [2, 25, 84, 56, 8, 70, 58, 78, 4, 46, 65, 29])

    selector = GridPartitioning(cells=3)
    selected_ids = selector.select(arr=coords, size=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [7, 55, 70, 57, 29, 91, 9, 65, 28, 11, 54, 88])


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
