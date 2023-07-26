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

"""Testing for the distance and similarity algorithms in the distance.py module."""

from DiverseSelector.distance import (compute_distance_matrix,
                                      pairwise_similarity_bit,
                                      )

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

# each row is a feature and each column is a molecule
sample1 = np.array([[4, 2, 6],
                    [4, 9, 6],
                    [2, 5, 0],
                    [2, 0, 9],
                    [5, 3, 0]])

# each row is a molecule and each column is a feature (scipy)
sample2 = np.array([[1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])

sample3 = np.array([[1, 4],
                    [3, 2]])

sample4 = np.array([[1, 0, 1],
                    [0, 1, 1]])


def test_compute_distance_matrix_builtin():
    """Testing the compute distance matrix with a built in metric."""
    sci_dist = compute_distance_matrix(sample2, "tanimoto")
    expected = np.array([[0, 0.6666667, 1, 1],
                         [0.6666667, 0, 1, 1],
                         [1, 1, 0, 1],
                         [1, 1, 1, 0]])
    assert_almost_equal(expected, sci_dist)


def test_compute_distance_matrix_invalid_metric():
    """Testing the compute distance matrix with an invalid metric."""
    assert_raises(ValueError, compute_distance_matrix, sample1, "fake_distance")


def test_tanimoto():
    """Testing the tanimoto function with predefined feature matrix."""
    tani = pairwise_similarity_bit(sample3, "tanimoto")
    expected = np.array([[1, (11 / 19)],
                         [(11 / 19), 1]])
    assert_equal(expected, tani)


def test_modifed_tanimoto():
    """Testing the modified tanimoto function with predefined feature matrix."""
    mod_tani = pairwise_similarity_bit(sample4, "modified_tanimoto")
    expceted = np.array([[1, (4 / 27)],
                         [(4 / 27), 1]])
    assert_equal(mod_tani, expceted)









