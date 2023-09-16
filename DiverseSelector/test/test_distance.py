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

"""Test distance.py Module."""


from DiverseSelector.distance import (
    pairwise_similarity_bit,
    tanimoto,
    modified_tanimoto,
)

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises


def test_pairwise_similarity_bit_raises():
    # check raised error for input feature matrix that is not 2D
    assert_raises(ValueError, pairwise_similarity_bit, np.random.random(5), "tanimoto")
    assert_raises(ValueError, pairwise_similarity_bit, np.random.random((2, 3, 4)), "tanimoto")
    # check raised error for not-available method
    assert_raises(ValueError, pairwise_similarity_bit, np.random.random((5, 1)), "tan")
    assert_raises(ValueError, pairwise_similarity_bit, np.random.random((5, 1)), tanimoto)


def test_tanimoto_raises():
    # check raised error when a or b is not 1D
    assert_raises(ValueError, tanimoto, np.random.random((1, 5)), np.random.random(5))
    assert_raises(ValueError, tanimoto, np.random.random(3), np.random.random((1, 4)))
    assert_raises(ValueError, tanimoto, np.random.random(4), np.random.random((3, 4)))
    assert_raises(ValueError, tanimoto, np.random.random((3, 3)), np.random.random((2, 3)))
    # check raised error when a and b don't have the same length
    assert_raises(ValueError, tanimoto, np.random.random(3), np.random.random(5))
    assert_raises(ValueError, tanimoto, np.random.random(20), np.random.random(10))


def test_tanimoto():
    """Test the tanimoto function on one pair of points."""
    a = np.array([2, 0, 1])
    b = np.array([2, 0, 0])
    # expected = (2*2 + 0*0 + 1*0) / (2**2 + 1 + 2**2 - 2*2)
    assert_equal(tanimoto(a, b), 4 / (5 + 4 - 4))


def test_tanimoto_bitstring():
    """Test the tanimoto function on one pair of points."""
    a = np.array([0, 0, 0, 1, 0, 1, 1])
    b = np.array([1, 1, 0, 0, 0, 1, 1])
    assert_equal(tanimoto(a, b), 2 / 5)


def test_tanimoto_matrix():
    """Testing the tanimoto function with predefined feature matrix."""
    x = np.array([[1, 4], [3, 2]])
    s = pairwise_similarity_bit(x, "tanimoto")
    expected = np.array([[1, (11 / 19)], [(11 / 19), 1]])
    assert_equal(s, expected)


def test_modified_tanimoto():
    a = np.array([1, 1, 0, 0, 1])
    b = np.array([0, 0, 0, 0, 1])
    expected = (1.6 / 9) + (1.4 / 6)
    assert_equal(modified_tanimoto(a, b), expected)


def test_modified_tanimoto_all_ones():
    """Test the modified tanimoto function when input is all '1' bits"""
    a = np.array([1, 1, 1, 1, 1])
    assert_equal(modified_tanimoto(a, a), 1)


def test_modified_tanimoto_all_zeroes():
    """Test the modified tanimoto function when input is all '0' bits"""
    a = np.zeros(5)
    assert_equal(modified_tanimoto(a, a), 1)


def test_modified_tanimoto_dimension_error():
    """Test modified tanimoto raises error when input has incorrect dimension."""
    a = np.zeros([7, 5])
    b = np.zeros(5)
    assert_raises(ValueError, modified_tanimoto, a, b)
    assert_raises(ValueError, modified_tanimoto, b, a)
    assert_raises(ValueError, modified_tanimoto, np.ones(3), np.ones(5))


def test_modified_tanimoto_matrix():
    """Testing the modified tanimoto function with predefined feature matrix."""
    x = np.array([[1, 0, 1], [0, 1, 1]])
    s = pairwise_similarity_bit(x, "modified_tanimoto")
    expceted = np.array([[1, (4 / 27)], [(4 / 27), 1]])
    assert_equal(s, expceted)
