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
from DiverseSelector.metric import ( 
                                    bit_tanimoto,
                                    distance_to_similarity,
                                    pairwise_similarity_bit,
                                    tanimoto,
                                    ComputeDistanceMatrix,
                                    )

from DiverseSelector.test.common import (
                                    bit_dice,
                                    euc_bit,
                                    bit_cosine,
                                    cosine,
                                    dice,
                                    )

import numpy as np
from numpy.testing import assert_equal


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
    # incomplet
    sci_dist = ComputeDistanceMatrix(sample1, "euclidean")
    selected = sci_dist.compute_distance()
    expected = 0
    assert_equal(expected, selected)


def test_Compute_Distance_Matrix_dice():
    # incomplet
    sci_dist = ComputeDistanceMatrix(sample4, "dice")
    selected = sci_dist.compute_distance()
    expected = np.array([[1, 0.5],
                        [0.5, 1]])
    assert_equal(expected, selected)


def test_Compute_Distance_Matrix_cosine():
    # incomplet
    sci_dist = ComputeDistanceMatrix(sample3, "cosine")
    selected = sci_dist.compute_distance()
    expected = distance_to_similarity(pairwise_similarity_bit(sample3 ,cosine), False)
    assert_equal(expected, selected)


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
