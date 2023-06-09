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

from DiverseSelector.distance import (bit_tanimoto,
                                      compute_distance_matrix,
                                      euc_bit,
                                      modified_tanimoto,
                                      pairwise_similarity_bit,
                                      tanimoto,
                                      nearest_average_tanimoto
                                      )

from DiverseSelector.utils import distance_to_similarity
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

#############################
# This group tests the compute_distance_matrix() function from distance.py


def test_compute_distance_matrix_sklearn():
    """Testing the compute distance matrix with a metric from sklearn."""
    sci_dist = compute_distance_matrix(sample3, "euclidean")
    expected = np.array([[0, 2.8284271],
                         [2.8284271, 0]])
    assert_almost_equal(expected, sci_dist)


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

#####################################################


def test_tanimoto_bit():
    """Testing the tanimoto function with predefined bit-string matrix."""
    tani = pairwise_similarity_bit(sample2, "bit_tanimoto")
    expected = np.array([[1, (1 / 3), 0, 0],
                         [(1 / 3), 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    assert_equal(expected, tani)


def test_tanimoto():
    """Testing the tanimoto function with predefined feature matrix."""
    tani = pairwise_similarity_bit(sample3, "tanimoto")
    expected = np.array([[1, (11 / 19)],
                         [(11 / 19), 1]])
    assert_equal(expected, tani)


def test_dist_to_simi():
    """Testing the distance to similarity function with predefined distance matrix."""
    dist = distance_to_similarity(sample3, dist=True)
    expected = np.array([[(1 / 2), (1 / 5)],
                         [(1 / 4), (1 / 3)]])
    assert_equal(dist, expected)


def test_modifed_tanimoto():
    """Testing the modified tanimoto function with predefined feature matrix."""
    mod_tani = pairwise_similarity_bit(sample4, "modified_tanimoto")
    expceted = np.array([[1, (4 / 27)],
                         [(4 / 27), 1]])
    assert_equal(mod_tani, expceted)


def test_nearest_average_tanimoto_bit():
    """Test the nearest_average_tanimoto function with binary input"""
    nat = nearest_average_tanimoto(sample2)
    shortest_tani = [0.3333333, 0.3333333, 0, 0]
    average = np.average(shortest_tani)
    assert_almost_equal(nat, average)


def test_nearest_average_tanimoto():
    """Test the nearest_average_tanimoto function with non-binary input"""
    nat = nearest_average_tanimoto(sample3)
    euc_dist = np.array([[0., 2.82842712],
                        [2.82842712, 0.]])
    all_tanimoto = np.array([[1, (11 / 19)],
                            [(11 / 19), 1]])
    shortest_tani = [(11/19), (11/19)]
    average = np.average(shortest_tani)
    assert_equal(nat, average)


# Bitstring Function Equivalence Testing
def test_bitstring_equivalence_tanimoto():
    bit_tani = pairwise_similarity_bit(sample2, "bit_tanimoto")
    tani = pairwise_similarity_bit(sample2, "tanimoto")
    assert_equal(bit_tani, tani)


def test_bitstring_equivalence_euc():
    bit_euc = compute_distance_matrix(sample2, "euclidean")
    euc = pairwise_similarity_bit(sample2, "euc_bit") - np.identity(len(sample2))
    assert_equal(bit_euc, euc)





