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

"""Testing for Utils.py."""


from DiverseSelector.utils import distance_to_similarity
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

# each row is a feature and each column is a molecule
sample1 = np.array([[4, 2, 6],
                    [4, 9, 6],
                    [2, 5, 0],
                    [2, 0, 9],
                    [5, 3, 0]])

# each row is a molecule and each colume is a feature (scipy)
sample2 = np.array([[1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])

sample3 = np.array([[1, 4],
                    [3, 2]])

sample4 = np.array([[1, 0, 1],
                    [0, 1, 1]])


def test_dist_to_simi():
    """Testing the distance to similarity function with predefined distance matrix."""
    dist = distance_to_similarity(sample3, dist=True)
    expceted = np.array([[(1 / 2), (1 / 5)],
                         [(1 / 4), (1 / 3)]])
    assert_equal(dist, expceted)

