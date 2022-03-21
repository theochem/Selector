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
from numpy.testing import assert_equal
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances


def test_minmax_selector1():
    """Testing the MinMax selection algorithm with predefined starting point."""
    syn_data, _ = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)
    arr_dist = pairwise_distances(syn_data)
    model = DissimilaritySelection(num_selected=10, arr_dist=arr_dist, random_seed=42)
    model.starting_idx = 0
    selected = model.select()
    assert_equal([0, 94, 3, 50, 64, 85, 93, 83, 34, 59], selected)
