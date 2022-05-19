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
from DiverseSelector.selectors import MaxMin, OptiSim
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
    assert_equal(selected_ids, [41, 34, 94, 85, 51, 50, 66, 78, 21, 64, 29, 83])

    selector = MaxMin()
    selected_ids = selector.select(arr=arr_dist, num_selected=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [85, 57, 41, 25, 9, 62, 29, 65, 81, 61, 60, 97])

def test_optisim():
    """Testing OptiSim class."""
    selector = OptiSim()
    selected_ids = selector.select(arr=coords_cluster, num_selected=12, labels=class_labels_cluster)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [2, 85, 86, 59, 1, 50, 66, 81, 0, 11, 33, 46])

    selector = OptiSim()
    selected_ids = selector.select(arr=coords, num_selected=12)
    # make sure all the selected indices are the same with expectation
    assert_equal(selected_ids, [0, 8, 25, 9, 21, 13, 37, 40, 65, 57, 18, 6])
