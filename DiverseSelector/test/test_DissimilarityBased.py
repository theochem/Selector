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

from collections import Counter

from DiverseSelector import DissimilaritySelection
from DiverseSelector.test.common import generate_synthetic_data
from numpy.testing import assert_equal


def test_minmax_selector_3_100():
    """Testing the MinMax selection algorithm with predefined starting point."""
    # in the function name:
    # 3 means that the number of clusters is 3
    # 100 means that the number of total data pints is 100
    _, class_labels, arr_dist = generate_synthetic_data(n_samples=100,
                                                        n_features=2,
                                                        n_clusters=3,
                                                        pairwise_dist=True,
                                                        metric="euclidean",
                                                        random_state=42)
    model = DissimilaritySelection(num_selected=12,
                                   arr_dist=arr_dist,
                                   random_seed=42)
    model.starting_idx = 0
    selected = model.select()

    # make sure all the selected indices are the same with expectation
    assert_equal([0, 94, 3, 50, 64, 85, 93, 83, 34, 59, 49, 72], selected)

    # make sure number of selected molecules is correct in reach cluster
    selected_labels_count = Counter(class_labels[selected])
    assert_equal(selected_labels_count[0], 4)
    assert_equal(selected_labels_count[1], 4)
    assert_equal(selected_labels_count[2], 4)
