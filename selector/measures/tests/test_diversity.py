# -*- coding: utf-8 -*-
#
# The Selector is a Python library of algorithms for selecting diverse
# subsets of data for machine-learning.
#
# Copyright (C) 2022-2024 The QC-Devs Community
#
# This file is part of Selector.
#
# Selector is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Selector is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --

"""Test Diversity Module."""
import warnings

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal, assert_raises, assert_warns

from selector.measures.diversity import (
    compute_diversity,
    explicit_diversity_index,
    gini_coefficient,
    hypersphere_overlap_of_subset,
    logdet,
    nearest_average_tanimoto,
    shannon_entropy,
    wdud,
)

# each row is a feature and each column is a molecule
sample1 = np.array([[4, 2, 6], [4, 9, 6], [2, 5, 0], [2, 0, 9], [5, 3, 0]])

# each row is a molecule and each column is a feature (scipy)
sample2 = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

sample3 = np.array([[1, 4], [3, 2]])

sample4 = np.array([[1, 0, 1], [0, 1, 1]])

sample5 = np.array([[0, 2, 4, 0], [1, 2, 4, 0], [2, 2, 4, 0]])

sample6 = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 0]])

sample7 = np.array([[1, 0, 1, 0] for _ in range(4)])

sample8 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])


def test_compute_diversity_specified():
    """Test compute diversity with a specified div_type."""
    comp_div = compute_diversity(sample6, "shannon_entropy", normalize=False, truncation=False)
    expected = 1.81
    assert round(comp_div, 2) == expected


def test_compute_diversity_hyperspheres():
    """Test compute diversity with two arguments for hypersphere_overlap method"""
    corner_pts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    centers_pts = np.array([[0.5, 0.5]] * (100 - 4))
    pts = np.vstack((corner_pts, centers_pts))

    comp_div = compute_diversity(pts, div_type="hypersphere_overlap", features=pts)
    # Expected = overlap + edge penalty
    expected = (100.0 * 96 * 95 * 0.5) + 2.0
    assert_almost_equal(comp_div, expected)


def test_compute_diversity_hypersphere_error():
    """Test compute diversity with hypersphere metric and no molecule library given."""
    assert_raises(ValueError, compute_diversity, sample5, "hypersphere_overlap")


def test_compute_diversity_edi():
    """Test compute diversity with explicit diversity index div_type"""
    z = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    cs = 1
    expected = 56.39551204
    actual = compute_diversity(z, "explicit_diversity_index", cs=cs)
    assert_almost_equal(expected, actual)


def test_compute_diversity_edi_no_cs_error():
    """Test compute diversity with explicit diversity index and no `cs` value given."""
    assert_raises(ValueError, compute_diversity, sample5, "explicit_diversity_index")


def test_compute_diversity_edi_zero_error():
    """Test compute diversity with explicit diversity index and `cs` = 0."""
    assert_raises(ValueError, compute_diversity, sample5, "explicit diversity index", cs=0)


def test_compute_diversity_invalid():
    """Test compute diversity with a non-supported div_type."""
    assert_raises(ValueError, compute_diversity, sample1, "diversity_type")


def test_logdet():
    """Test the log determinant function with predefined subset matrix."""
    sel = logdet(sample3)
    expected = np.log(131)
    assert_almost_equal(sel, expected)


def test_logdet_non_square_matrix():
    """Test the log determinant function with a rectangular matrix."""
    sel = logdet(sample4)
    expected = np.log(8)
    assert_almost_equal(sel, expected)


def test_shannon_entropy():
    """Test the shannon entropy function with example from the original paper."""

    # example taken from figure 1 of 10.1021/ci900159f
    x1 = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 0]])
    expected = 1.81
    assert round(shannon_entropy(x1, normalize=False, truncation=False), 2) == expected

    x2 = np.vstack((x1, [1, 1, 1, 0]))
    expected = 1.94
    assert round(shannon_entropy(x2, normalize=False, truncation=False), 2) == expected

    x3 = np.vstack((x1, [0, 1, 0, 1]))
    expected = 3.39
    assert round(shannon_entropy(x3, normalize=False, truncation=False), 2) == expected


def test_shannon_entropy_normalize():
    """Test the shannon entropy function with normalization."""
    x1 = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 0]])
    expected = 1.81 / (x1.shape[1] * np.log2(2) / 2)
    assert_almost_equal(
        actual=shannon_entropy(x1, normalize=True, truncation=False),
        desired=expected,
        decimal=2,
    )


def test_shannon_entropy_warning():
    """Test the shannon entropy function gives warning when normalization is True and truncation is True."""
    x1 = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 0]])
    with pytest.warns(UserWarning):
        shannon_entropy(x1, normalize=True, truncation=True)


def test_shannon_entropy_binary_error():
    """Test the shannon entropy function raises error with a non binary matrix."""
    assert_raises(ValueError, shannon_entropy, sample5)


def test_explicit_diversity_index():
    """Test the explicit diversity index function."""
    z = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    cs = 1
    nc = 3
    sdi = 0.75 / 0.7332902012
    cr = -1 * 0.4771212547
    edi = 0.5456661753 * 0.7071067811865476
    edi_scaled = 56.395512045413
    value = explicit_diversity_index(z, cs)
    assert_almost_equal(value, edi_scaled, decimal=8)


def test_wdud_uniform():
    """Test wdud when a feature has uniform distribution."""
    uni = np.arange(0, 50000)[:, None]
    wdud_val = wdud(uni)
    expected = 0
    assert_almost_equal(wdud_val, expected, decimal=4)


def test_wdud_repeat_yi():
    """Test wdud when a feature has multiple identical values."""
    dist = np.array([[0, 0.5, 0.5, 0.75, 1]]).T
    wdud_val = wdud(dist)
    # calculated using wolfram alpha:
    expected = 0.065 + 0.01625 + 0.02125
    assert_almost_equal(wdud_val, expected, decimal=4)


def test_wdud_mult_features():
    """Test wdud when there are multiple features per molecule."""
    dist = np.array(
        [
            [0, 0.5, 0.5, 0.75, 1],
            [0, 0.5, 0.5, 0.75, 1],
            [0, 0.5, 0.5, 0.75, 1],
            [0, 0.5, 0.5, 0.75, 1],
        ]
    ).T
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
    assert_raises(ValueError, wdud, sample8)


def test_wdud_warning_normalization():
    """Test wdud method gives warning when normalization fails."""
    warning_message = (
        "Some of the features are constant which will cause the normalization to fail. "
        + "Now removing them."
    )
    with pytest.warns() as record:
        wdud(sample6)

    # check that the message matches
    assert record[0].message.args[0] == warning_message


def test_hypersphere_overlap_of_subset_with_only_corners_and_center():
    """Test the hypersphere overlap method with predefined matrix."""
    corner_pts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
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
    assert_raises(ValueError, hypersphere_overlap_of_subset, sample7, sample7)


def test_hypersphere_radius_warning():
    """Test the hypersphere overlap method gives warning when radius is too large."""
    corner_pts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
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
    assert_raises(ValueError, gini_coefficient, np.array([[7, 0], [2, 1]]))


def test_gini_coefficient_dimension_error():
    """Test Gini coefficient error when input has incorrect dimensions."""
    assert_raises(ValueError, gini_coefficient, np.array([1, 0, 0, 0]))


def test_gini_coefficient_of_most_diverse_set():
    """Test Gini coefficient of the most diverse set."""
    #  Finger-prints where one feature has more `wealth` than all others.
    #  Note: Transpose is done so one column has all ones.
    finger_prints = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
        ]
        + [[0, 0, 0, 0, 0, 0, 0]] * 100000
    ).T
    result = gini_coefficient(finger_prints)
    # Since they are all the same, then gini coefficient should be zero.
    assert_almost_equal(result, 1.0, decimal=4)


def test_gini_coefficient_with_alternative_definition():
    """Test Gini coefficient with alternative definition."""
    # Finger-prints where they are all different
    numb_features = 4
    finger_prints = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    result = gini_coefficient(finger_prints)

    # Alternative definition from wikipedia
    b = numb_features + 1
    desired = (
        numb_features + 1 - 2 * ((b - 1) + (b - 2) * 2 + (b - 3) * 3 + (b - 4) * 4) / (10)
    ) / numb_features
    assert_almost_equal(result, desired)


def test_nearest_average_tanimoto_bit():
    """Test the nearest_average_tanimoto function with binary input."""
    nat = nearest_average_tanimoto(sample2)
    shortest_tani = [0.3333333, 0.3333333, 0, 0]
    average = np.average(shortest_tani)
    assert_almost_equal(nat, average)


def test_nearest_average_tanimoto():
    """Test the nearest_average_tanimoto function with non-binary input."""
    nat = nearest_average_tanimoto(sample3)
    shortest_tani = [(11 / 19), (11 / 19)]
    average = np.average(shortest_tani)
    assert_equal(nat, average)


def test_nearest_average_tanimoto_3_x_3():
    """Testpyth the nearest_average_tanimoto function with a 3x3 matrix."""
    # all unequal distances b/w points
    x = np.array([[0, 1, 2], [3, 4, 5], [4, 5, 6]])
    nat_x = nearest_average_tanimoto(x)
    avg_x = 0.749718574108818
    assert_equal(nat_x, avg_x)
    # one point equidistant from the other two
    y = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    nat_y = nearest_average_tanimoto(y)
    avg_y = 0.4813295920569825
    assert_equal(nat_y, avg_y)
    # all points equidistant
    z = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    nat_z = nearest_average_tanimoto(z)
    avg_z = 0.25
    assert_equal(nat_z, avg_z)


def test_nearest_average_tanimoto_nonsquare():
    """Test the nearest_average_tanimoto function with non-binary input"""
    x = np.array([[3.5, 4.0, 10.5, 0.5], [1.25, 4.0, 7.0, 0.1], [0.0, 0.0, 0.0, 0.0]])
    # nearest neighbor of sample 0, 1, and 2 are sample 1, 0, and 1, respectively.
    expected = np.average(
        [
            np.sum(x[0] * x[1]) / (np.sum(x[0] ** 2) + np.sum(x[1] ** 2) - np.sum(x[0] * x[1])),
            np.sum(x[1] * x[0]) / (np.sum(x[1] ** 2) + np.sum(x[0] ** 2) - np.sum(x[1] * x[0])),
            np.sum(x[2] * x[1]) / (np.sum(x[2] ** 2) + np.sum(x[1] ** 2) - np.sum(x[2] * x[1])),
        ]
    )
    assert_equal(nearest_average_tanimoto(x), expected)
