# -*- coding: utf-8 -*-
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

"""Testing for the dissimilarity-based selection algorithms."""

from DiverseSelector import DissimilaritySelection
from DiverseSelector.test.common import generate_synthetic_data
from numpy.testing import assert_equal

coords, class_labels, arr_dist = generate_synthetic_data(n_samples=100,
                                                         n_features=2,
                                                         n_clusters=1,
                                                         pairwise_dist=True,
                                                         metric="euclidean",
                                                         random_state=42)


def test_brute_strength_maxmin():
    """Testing brutestrength algorithm with maxmin."""
    selector = DissimilaritySelection(num_selected=12,
                                      arr_dist=arr_dist,
                                      random_seed=42)
    selector.starting_idx = 0
    selector.features = coords
    selected_ids = selector.select("brute_strength")

    # make sure all the selected indices are the same with expectation
    assert_equal([0, 57, 95, 41, 67, 26, 3, 16, 12, 6, 62, 48], selected_ids)


def test_brutestrength_maxsum():
    """Testing brutestrength algorithm with maxsum."""
    selector = DissimilaritySelection(num_selected=12,
                                      arr_dist=arr_dist,
                                      random_seed=42,
                                      method="maxsum")
    selector.starting_idx = 0
    selector.features = coords
    selected_ids = selector.select()

    # make sure all the selected indices are the same with expectation
    assert_equal([0, 57, 25, 41, 95, 9, 8, 21, 13, 68, 37, 54], selected_ids)


def test_gridpartitioning_equisized_independent():
    """Testing gridpartitioning algorithm with equisized independent partitioning method."""
    selector = DissimilaritySelection(num_selected=12,
                                      arr_dist=arr_dist,
                                      random_seed=42)
    selector.starting_idx = 0
    selector.features = coords
    selected_ids = selector.select("grid_partitioning")

    # make sure all the selected indices are the same with expectation
    assert_equal([15, 87, 70, 66, 49, 68, 8, 22, 10, 13,
                  19, 44, 76, 72, 25, 84, 73, 57, 65, 86], selected_ids)


def test_gridpartitioning_equisized_dependent():
    """Testing gridpartitioning algorithm with equisized dependent partitioning method."""
    selector = DissimilaritySelection(num_selected=12,
                                      arr_dist=arr_dist,
                                      random_seed=42,
                                      grid_method="equisized_dependent")
    selector.starting_idx = 0
    selector.features = coords
    selected_ids = selector.select("grid_partitioning")

    # make sure all the selected indices are the same with expectation
    assert_equal([0, 87, 68, 59, 50, 79, 4, 41, 30, 33, 71,
                  98, 73, 80, 65, 19, 10, 25, 55, 54, 37, 57, 86], selected_ids)


def test_sphereexclusion():
    """Testing sphereexclusion algorithm."""
    selector = DissimilaritySelection(num_selected=12,
                                      arr_dist=arr_dist,
                                      random_seed=42)
    selector.starting_idx = 0
    selector.features = coords
    selected_ids = selector.select("sphere_exclusion")

    # make sure all the selected indices are the same with expectation
    assert_equal([17, 31, 90, 6, 12, 76, 26, 81, 2, 14, 57], selected_ids)


def test_optisim():
    """Testing optisim algorithm."""
    selector = DissimilaritySelection(num_selected=12,
                                      arr_dist=arr_dist,
                                      random_seed=42)
    selector.starting_idx = 0
    selector.features = coords
    selected_ids = selector.select("optisim")

    # make sure all the selected indices are the same with expectation
    assert_equal([0, 13, 21, 9, 8, 18, 57, 39, 65, 25], selected_ids)
