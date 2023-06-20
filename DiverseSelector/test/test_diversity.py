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
                                       compute_diversity,
                                       entropy,
                                       gini_coefficient,
                                       logdet,
                                       shannon_entropy,
                                       hypersphere_overlap_of_subset,
                                       wdud,
                                       )
from DiverseSelector.utils import distance_to_similarity
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises, assert_warns

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

sample5 = np.array([[0, 2, 4, 0],
                    [1, 2, 4, 0],
                    [2, 2, 4, 0]])


def test_compute_diversity_default():
    """Test compute diversity with default div_type."""
    comp_div = compute_diversity(sample4)
    expected = (2/3)
    assert_almost_equal(comp_div, expected)


def test_compute_diversity_specified():
    """Test compute diversity with a specified div_type."""
    comp_div = compute_diversity(sample4, "shannon_entropy")
    expected = 0.301029995
    assert_almost_equal(comp_div, expected)


def test_compute_diversity_hyperspheres():
    """Test compute diversity with two arguments for hypersphere_overlap method"""
    corner_pts = np.array([[0.0, 0.0],
                           [0.0, 1.0],
                           [1.0, 0.0],
                           [1.0, 1.0]])
    centers_pts = np.array([[0.5, 0.5]] * (100 - 4))
    pts = np.vstack((corner_pts, centers_pts))

    comp_div = compute_diversity(pts, "hypersphere_overlap_of_subset", pts)
    # Expected = overlap + edge penalty
    expected = (100.0 * 96 * 95 * 0.5) + 2.0
    assert_almost_equal(comp_div, expected)


def test_compute_diversity_invalid():
    """Test compute diversity with a non-supported div_type."""
    assert_raises(ValueError, compute_diversity, sample1, "diversity_type")


def test_entropy():
    """Test the entropy function with predefined matrix."""
    ent = entropy(sample4)
    expected = (2 / 3)
    assert_almost_equal(ent, expected)


def test_entropy_conversion():
    """Test the entropy function with matrix that is not in bit form."""
    ent = entropy(sample3)
    expected = 0
    assert_almost_equal(ent, expected)


def test_entropy_value_error():
    """Test the entropy function with a matrix that causes a value error"""
    assert_raises(ValueError, entropy, sample5)


def test_logdet():
    """Test the log determinant function with predefined subset matrix."""
    sel = logdet(sample3)
    expected = np.log10(131)
    assert_almost_equal(sel, expected)


def test_logdet_non_square_matrix():
    """Test the log determinant function with a rectangular matrix."""
    sel = logdet(sample4)
    expected = np.log10(8)
    assert_almost_equal(sel, expected)


def test_shannon_entropy():
    """Test the shannon entropy function with predefined matrix."""
    selected = shannon_entropy(sample4)
    # expected = -log10(1/2)
    expected = 0.301029995
    assert_almost_equal(selected, expected)


def test_shannon_entropy_error():
    """Test the shannon entropy function raises error with matrix with invalid feature."""
    assert_raises(ValueError, shannon_entropy, sample5)


def test_wdud_uniform():
    """Test wdud when a feature has uniform distribution."""
    uni = np.arange(0, 50000)[:, None]
    wdud_val = wdud(uni)
    expected = 0
    assert_almost_equal(wdud_val, expected, decimal=4)


def test_wdud_repeat_yi():
    """Test wdud when a feature has multiple identical values."""
    dist = np.array([[0,0.5,0.5,0.75,1]]).T
    wdud_val = wdud(dist)
    # calculated using wolfram alpha:
    expected = 0.065 + 0.01625 + 0.02125
    assert_almost_equal(wdud_val, expected, decimal=4)


def test_wdud_mult_features():
    """Test wdud when there are multiple features per molecule."""
    dist = np.array([[0, 0.5, 0.5, 0.75, 1],
                     [0, 0.5, 0.5, 0.75, 1],
                     [0, 0.5, 0.5, 0.75, 1],
                     [0, 0.5, 0.5, 0.75, 1]]).T
    wdud_val = wdud(dist)
    # calculated using wolfram alpha:
    expected = 0.065 + 0.01625 + 0.02125
    assert_almost_equal(wdud_val, expected, decimal=4)


def test_wdud_dimension_error():
    """Test wdud method raises error when input has incorrect dimensions."""
    arr = np.zeros((2, 2, 2))
    assert_raises(ValueError, wdud, arr)


def test_wdud_normalization_error():
    """Test wdud method raises error when normalization fails."""
    assert_raises(ValueError, wdud, sample5)


def test_hypersphere_overlap_of_subset_with_only_corners_and_center():
    """Test the hypersphere overlap method with predefined matrix."""
    corner_pts = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    # Many duplicate pts cause r_0 to be much smaller than 1.0,
    #   which is required due to normalization of the feature space
    centers_pts = np.array([[0.5, 0.5]] * (100 - 4))
    pts = np.vstack((corner_pts, centers_pts))

    # Overlap should be all coming from the centers
    expected_overlap = 100.0 * 96 * 95 * 0.5
    # The edge penalty should all be from the corner pts
    lam = 1.0 / 2.0  # Default lambda chosen from paper.
    expected_edge = lam * 4.0
    expected = expected_overlap + expected_edge
    true = hypersphere_overlap_of_subset(pts, pts)
    assert_almost_equal(true, expected)


def test_hypersphere_normalization_error():
    """Test the hypersphere overlap method raises error when normalization fails."""
    assert_raises(ValueError, hypersphere_overlap_of_subset, sample5, sample5)


def test_hypersphere_radius_warning():
    """Test the hypersphere overlap method gives warning when radius is too large."""
    corner_pts = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    assert_warns(Warning, hypersphere_overlap_of_subset, corner_pts, corner_pts)


def test_gini_coefficient_of_non_diverse_set():
    """Test Gini coefficient of the least diverse set. Expected return is zero."""
    # Finger-prints where columns are all the same
    numb_molecules = 5
    numb_features = 10
    # Transpose so that the columns are all the same, note first made the rows all same
    single_fingerprint = list(np.random.choice([0, 1], size=(numb_features,)))
    finger_prints = np.array([single_fingerprint] * numb_molecules).T

    result = gini_coefficient(finger_prints)
    # Since they are all the same, then gini coefficient should be zero.
    assert_almost_equal(result, 0.0, decimal=8)


def test_gini_coefficient_non_binary_error():
    """Test Gini coefficient error when input is not binary."""
    assert_raises(ValueError, gini_coefficient, np.array([[1, 2], [7, 1]]))


def test_gini_coefficient_dimension_error():
    """Test Gini coefficient error when input has incorrect dimensions."""
    assert_raises(ValueError, gini_coefficient, np.array([1, 0, 0, 0]))


def test_gini_coefficient_of_most_diverse_set():
    """Test Gini coefficient of the most diverse set."""
    #  Finger-prints where one feature has more `wealth` than all others.
    #  Note: Transpose is done so one column has all ones.
    finger_prints = np.array([
                                 [1, 1, 1, 1, 1, 1, 1],

                             ] + [[0, 0, 0, 0, 0, 0, 0]] * 100000).T
    result = gini_coefficient(finger_prints)
    # Since they are all the same, then gini coefficient should be zero.
    assert_almost_equal(result, 1.0, decimal=4)


def test_gini_coefficient_with_alternative_definition():
    """Test Gini coefficient with alternative definition."""
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
