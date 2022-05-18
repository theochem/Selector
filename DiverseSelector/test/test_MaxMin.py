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

"""Testing for the MaxMin selection algorithms."""

from DiverseSelector.test.common import generate_synthetic_data
from DiverseSelector.Selectors import MaxMin
from numpy.testing import assert_equal

coords, class_labels, arr_dist = generate_synthetic_data(n_samples=100,
                                                         n_features=2,
                                                         n_clusters=1,
                                                         pairwise_dist=True,
                                                         metric="euclidean",
                                                         random_state=42)

coords_cluster, class_labels_cluster, arr_dist_cluster = generate_synthetic_data(n_samples=100,
                                                                                 n_features=2,
                                                                                 n_clusters=3,
                                                                                 pairwise_dist=True,
                                                                                 metric="euclidean",
                                                                                 random_state=42)


def test_maxmin():
    """Testing the MinMax class."""
    selector = MaxMin()
    selected_ids = selector.select(arr=arr_dist_cluster, num_selected=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [34, 59, 94, 73, 50, 3, 93, 49, 64, 0, 83, 72])

    selector = MaxMin()
    selected_ids = selector.select(arr=arr_dist, num_selected=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [57, 25, 68, 8, 53, 9, 95, 60, 76, 66, 69, 20])
