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

"""Testing for the diversity algorithms in the diversity.py module."""

from DiverseSelector.diversity import (
                                       # compute_diversity_matrix,
                                       entropy,
                                       gini_coefficient,
                                       logdet,
                                       shannon_entropy,
                                       total_diversity_volume,
                                       # wdud,
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

# each row is a molecule and each colume is a feature (scipy)
sample2 = np.array([[1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])

sample3 = np.array([[1, 4],
                    [3, 2]])

sample4 = np.array([[1, 0, 1],
                    [0, 1, 1]])


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

# todo: implement Wasserstein test
def test_wdud():
    """Testing the Wasserstein Distance to Uniform Distribution (WDUD) with predefined matrix ."""
    # incomplete
    # selected = wdud(sample3)
    # expected = (2 / 3)
    # assert_equal(expected, selected)
    pass


def test_total_diversity_volume():
    """Testing the total diversity volume method with predefined matrix."""
    selected = total_diversity_volume(sample3)
    expected = 2
    assert_almost_equal(selected, expected)


def test_gini_coefficient_of_non_diverse_set():
    r"""Test Gini coefficient of the worst diverse set is zero."""
    # Finger-prints where columns are all the same
    numb_molecules = 5
    numb_features = 10
    # Transpose so that the columns are all the same, note first made the rows all same
    single_fingerprint = list(np.random.choice([0, 1], size=(numb_features,)))
    finger_prints = np.array([single_fingerprint] * numb_molecules).T

    result = gini_coefficient(finger_prints)
    # Since they are all the same, then gini coefficient should be zero.
    assert_almost_equal(result, 0.0, decimal=8)

    # Test raises as well.
    assert_raises(ValueError, gini_coefficient, np.array([[1, 2], [0, 1]]))
    assert_raises(ValueError, gini_coefficient, np.array([1, 0, 0, 0]))


def test_gini_coefficient_of_most_diverse_set():
    r"""Test Gini coefficient of the most diverse set."""
    #  Finger-prints where one feature has more `wealth` than all others.
    #  Note transpose is done so one column has all ones.
    finger_prints = np.array([
                                 [1, 1, 1, 1, 1, 1, 1],

                             ] + [[0, 0, 0, 0, 0, 0, 0]] * 100000).T
    result = gini_coefficient(finger_prints)
    # Since they are all the same, then gini coefficient should be zero.
    assert_almost_equal(result, 1.0, decimal=4)


def test_gini_coefficient_with_alternative_definition():
    r"""Test Gini coefficient with alternative definition."""
    # Finger-prints where they are all different
    numb_features = 4
    finger_prints = np.array([
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ])
    result = gini_coefficient(finger_prints)

    # Alternative definition from wikipedia
    b = numb_features + 1
    desired = (
                      numb_features + 1 - 2 * (
                          (b - 1) + (b - 2) * 2 + (b - 3) * 3 + (b - 4) * 4) / (10)
              ) / numb_features
    assert_almost_equal(result, desired)
