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
from DiverseSelector.metric import  (
                                    bit_tanimoto,
                                    distance_to_similarity,
                                    pairwise_similarity_bit,
                                    tanimoto,
                                    ComputeDistanceMatrix,
                                    modified_tanimoto,
                                    entropy,
                                    logdet,
                                    total_diversity_Volume,
                                    shannon_entropy,
                                    wdud,
                                    )

from DiverseSelector.test.common import (
                                    bit_dice,
                                    euc_bit,
                                    bit_cosine,
                                    cosine,
                                    dice,
                                    )

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal


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


def test_Compute_Distance_Matrix_Euc_bit():
    sci_dist = ComputeDistanceMatrix(sample2, "euclidean")
    selected = sci_dist.compute_distance()
    expected = pairwise_similarity_bit(sample2, euc_bit) - np.identity(len(sample2))
    assert_equal(expected, selected)


def test_Compute_Distance_Matrix_Euc():
    sci_dist = ComputeDistanceMatrix(sample3, "euclidean")
    selected = sci_dist.compute_distance()
    expected = np.array([[0, 2.8284271],
                        [2.8284271, 0]])
    assert_almost_equal(expected, selected)


def test_tanimoto_bit():
    tani = pairwise_similarity_bit(sample2, bit_tanimoto)
    expceted = np.array([[1, (1 / 3), 0, 0],
                        [(1 / 3), 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    assert_equal(expceted, tani)


def test_tanimoto():
    tani = pairwise_similarity_bit(sample3, tanimoto)
    expceted = np.array([[1, (11 / 19)],
                        [(11 / 19), 1]])
    assert_equal(expceted, tani)


def test_dist_to_simi():
    dist = distance_to_similarity(sample3, dist=True)
    expceted = np.array([[(1/2), (1 / 5)],
                        [(1 / 4), (1 / 3)]])
    assert_equal(dist, expceted)


def test_modifed_tanimoto():
    # answer is negative 
    mod_tani = pairwise_similarity_bit(sample4, modified_tanimoto)
    expceted = np.array([[1, (4 / 27)],
                        [(4 / 27), 1]])
    assert_equal(mod_tani, expceted)


def test_entropy():
    ent = entropy(sample4) 
    expected = (2 / 3)
    assert_almost_equal(ent, expected)


def test_logdet():
    sel = logdet(sample3)
    expected = np.log10(131)
    assert_almost_equal(sel, expected)


def test_shannon_entropy():
    selected = shannon_entropy(sample4)
    expected = 0.301029995
    assert_almost_equal(selected, expected)


def test_wdud():
    # incomplet
    selected = wdud(sample3)
    expected = (2 / 3)
    assert_equal(expected, selected)


def test_total_diversity_volume():
    selected = total_diversity_Volume(sample3)
    expected = 2
    assert_almost_equal(selected, expected)
