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

"""Testing for the distance, similarity, and diversity algorithms."""

# todo: fix this later
# noqa: F401
from DiverseSelector.metric import (bit_tanimoto,
                                    ComputeDistanceMatrix,
                                    distance_to_similarity,
                                    entropy,
                                    euc_bit,
                                    logdet,
                                    modified_tanimoto,
                                    pairwise_similarity_bit,
                                    shannon_entropy,
                                    tanimoto,
                                    total_diversity_volume,
                                    # wdud
                                    )
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

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


def test_compute_distance_matrix_euc_bit():
    """Testing the euclidean distance function with predefined feature matrix."""
    sci_dist = ComputeDistanceMatrix(sample2, "euclidean")
    selected = sci_dist.compute_distance()
    expected = pairwise_similarity_bit(sample2, euc_bit) - np.identity(len(sample2))
    assert_equal(expected, selected)


def test_compute_distance_matrix_euc():
    """Testing the euclidean distance function with predefined bit-string matrix."""
    sci_dist = ComputeDistanceMatrix(sample3, "euclidean")
    selected = sci_dist.compute_distance()
    expected = np.array([[0, 2.8284271],
                        [2.8284271, 0]])
    assert_almost_equal(expected, selected)


def test_tanimoto_bit():
    """Testing the tanimoto function with predefined bit-string matrix."""
    tani = pairwise_similarity_bit(sample2, bit_tanimoto)
    expceted = np.array([[1, (1 / 3), 0, 0],
                        [(1 / 3), 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    assert_equal(expceted, tani)


def test_tanimoto():
    """Testing the tanimoto function with predefined feature matrix."""
    tani = pairwise_similarity_bit(sample3, tanimoto)
    expceted = np.array([[1, (11 / 19)],
                        [(11 / 19), 1]])
    assert_equal(expceted, tani)


def test_dist_to_simi():
    """Testing the distance to similarity function with predefined distance matrix."""
    dist = distance_to_similarity(sample3, dist=True)
    expceted = np.array([[(1 / 2), (1 / 5)],
                        [(1 / 4), (1 / 3)]])
    assert_equal(dist, expceted)


def test_modifed_tanimoto():
    """Testing the modified tanimoto function with predefined feature matrix."""
    mod_tani = pairwise_similarity_bit(sample4, modified_tanimoto)
    expceted = np.array([[1, (4 / 27)],
                        [(4 / 27), 1]])
    assert_equal(mod_tani, expceted)


def test_entropy():
    """Testing the entropy function with predefined matrix."""
    ent = entropy(sample4)
    expected = (2 / 3)
    assert_almost_equal(ent, expected)


def test_logdet():
    """Testing the log determinant function with predefined subset matrix."""
    sel = logdet(sample3)
    expected = np.log10(131)
    assert_almost_equal(sel, expected)


def test_shannon_entropy():
    """Testing the shannon entropy function with predefined matrix."""
    selected = shannon_entropy(sample4)
    expected = 0.301029995
    assert_almost_equal(selected, expected)


def test_wdud():
    """Testing the Wasserstein Distance to Uniform Distribution (WDUD) with predefined matrix ."""
    # incomplet
    # selected = wdud(sample3)
    # expected = (2 / 3)
    # assert_equal(expected, selected)
    pass


def test_total_diversity_volume():
    """Testing the total diversity volume method with predefined matrix."""
    selected = total_diversity_volume(sample3)
    expected = 2
    assert_almost_equal(selected, expected)
