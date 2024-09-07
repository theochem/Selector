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
"""Test Converter Module."""

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

from selector.measures import converter as cv

# Tests for variations on input `x` for sim_to_dist()


def test_sim_2_dist_float_int():
    """Test similarity to distance input handling when input is a float or int."""
    expected_1 = 0.25
    int_out = cv.sim_to_dist(4, "reciprocal")
    assert_equal(int_out, expected_1)
    expected_2 = 2
    float_out = cv.sim_to_dist(0.5, "reciprocal")
    assert_equal(float_out, expected_2)


def test_sim_2_dist_array_dimension_error():
    """Test sim to dist function with incorrect input dimensions for `x`."""
    assert_raises(ValueError, cv.sim_to_dist, np.ones([2, 2, 2]), "reciprocal")


def test_sim_2_dist_1d_metric_error():
    """Test sim to dist function with an invalid metric for 1D arrays."""
    assert_raises(ValueError, cv.sim_to_dist, np.ones(5), "gravity")
    assert_raises(ValueError, cv.sim_to_dist, np.ones(5), "co-occurrence")


# Tests for variations on input `metric` for sim_to_dist()


def test_sim_2_dist():
    """Test similarity to distance method with specified metric."""
    x = np.array([[1, 0.2, 0.5], [0.2, 1, 0.25], [0.5, 0.25, 1]])
    expected = np.array([[0.20, 1, 0.70], [1, 0.20, 0.95], [0.70, 0.95, 0.20]])
    actual = cv.sim_to_dist(x, "reverse")
    assert_almost_equal(actual, expected, decimal=10)


def test_sim_2_dist_frequency():
    """Test similarity to distance method with a frequency metric."""
    x = np.array([[4, 9, 1], [9, 1, 25], [1, 25, 16]])
    expected = np.array([[(1 / 2), (1 / 3), 1], [(1 / 3), 1, (1 / 5)], [1, (1 / 5), (1 / 4)]])
    actual = cv.sim_to_dist(x, "transition")
    assert_almost_equal(actual, expected, decimal=10)


def test_sim_2_dist_frequency_error():
    """Test similarity to distance method with a frequency metric and incorrect input."""
    # zeroes in the frequency matrix
    x = np.array([[0, 9, 1], [9, 1, 25], [1, 25, 0]])
    assert_raises(ValueError, cv.sim_to_dist, x, "gravity")
    # negatives in the frequency matrix
    y = np.array([[1, -9, 1], [9, 1, -25], [1, 25, 16]])
    assert_raises(ValueError, cv.sim_to_dist, x, "gravity")


def test_sim_2_dist_membership():
    """Test similarity to distance method with the membership metric."""
    # x = np.array([[(1 / 2), (1 / 5)], [(1 / 4), (1 / 3)]])
    x = np.array([[1, 1 / 5, 1 / 3], [1 / 5, 1, 4 / 5], [1 / 3, 4 / 5, 1]])
    expected = np.array([[0, 4 / 5, 2 / 3], [4 / 5, 0, 1 / 5], [2 / 3, 1 / 5, 0]])
    actual = cv.sim_to_dist(x, "membership")
    assert_almost_equal(actual, expected, decimal=10)


def test_sim_2_dist_membership_error():
    """Test similarity to distance method with the membership metric when there is an input error."""
    x = np.array([[1, 0, -7], [0, 1, 3], [-7, 3, 1]])
    assert_raises(ValueError, cv.sim_to_dist, x, "membership")


def test_sim_2_dist_invalid_metric():
    """Test similarity to distance method with an unsupported metric."""
    assert_raises(ValueError, cv.sim_to_dist, np.ones(5), "testing")


def test_sim_2_dist_non_symmetric():
    """Test the invalid 2D symmetric matrix error."""
    x = np.array([[1, 2], [4, 5]])
    assert_raises(ValueError, cv.sim_to_dist, x, "reverse")


# Tests for individual metrics


def test_reverse():
    """Test the reverse function for similarity to distance conversion."""
    x = np.array([[3, 1, 1], [1, 3, 0], [1, 0, 3]])
    expected = np.array([[0, 2, 2], [2, 0, 3], [2, 3, 0]])
    actual = cv.reverse(x)
    assert_equal(actual, expected)


def test_reciprocal():
    """Test the reverse function for similarity to distance conversion."""
    x = np.array([[1, 0.25, 0.40], [0.25, 1, 0.625], [0.40, 0.625, 1]])
    expected = np.array([[1, 4, 2.5], [4, 1, 1.6], [2.5, 1.6, 1]])
    actual = cv.reciprocal(x)
    assert_equal(actual, expected)


def test_reciprocal_error():
    """Test the reverse function with incorrect input values."""
    # zero value for similarity (causes divide by zero issues)
    x = np.array([[0, 4], [3, 2]])
    assert_raises(ValueError, cv.reciprocal, x)
    # negative value for similarity (distance cannot be negative)
    y = np.array([[1, -4], [3, 2]])
    assert_raises(ValueError, cv.reciprocal, y)


def test_exponential():
    """Test the exponential function for similarity to distance conversion."""
    x = np.array([[1, 0.25, 0.40], [0.25, 1, 0.625], [0.40, 0.625, 1]])
    expected = np.array(
        [
            [0, 1.38629436112, 0.91629073187],
            [1.38629436112, 0, 0.47000362924],
            [0.91629073187, 0.47000362924, 0],
        ]
    )
    actual = cv.exponential(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_exponential_error():
    """Test the exponential function when max similarity is zero."""
    x = np.zeros((4, 4))
    assert_raises(ValueError, cv.exponential, x)


def test_gaussian():
    """Test the gaussian function for similarity to distance conversion."""
    x = np.array([[1, 0.25, 0.40], [0.25, 1, 0.625], [0.40, 0.625, 1]])
    expected = np.array(
        [
            [0, 1.17741002252, 0.95723076208],
            [1.17741002252, 0, 0.68556810693],
            [0.95723076208, 0.68556810693, 0],
        ]
    )
    actual = cv.gaussian(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_gaussian_error():
    """Test the gaussian function when max similarity is zero."""
    x = np.zeros((4, 4))
    assert_raises(ValueError, cv.gaussian, x)


def test_correlation():
    """Test the correlation to distance conversion function."""
    x = np.array([[1, 0.5, 0.2], [0.5, 1, -0.2], [0.2, -0.2, 1]])
    # expected = sqrt(1-x)
    expected = np.array(
        [
            [0, 0.70710678118, 0.894427191],
            [0.70710678118, 0, 1.09544511501],
            [0.894427191, 1.09544511501, 0],
        ]
    )
    actual = cv.correlation(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_correlation_error():
    """Test the correlation function with an out of bounds array."""
    x = np.array([[1, 0, -7], [0, 1, 3], [-7, 3, 1]])
    assert_raises(ValueError, cv.correlation, x)


def test_transition():
    """Test the transition function for frequency to distance conversion."""
    x = np.array([[4, 9, 1], [9, 1, 25], [1, 25, 16]])
    expected = np.array([[(1 / 2), (1 / 3), 1], [(1 / 3), 1, (1 / 5)], [1, (1 / 5), (1 / 4)]])

    actual = cv.transition(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_co_occurrence():
    """Test the co-occurrence conversion function."""
    x = np.array([[1, 2, 3], [2, 1, 3], [3, 3, 1]])
    expected = np.array(
        [
            [1 / (19 / 121 + 1), 1 / (38 / 121 + 1), 1 / (57 / 121 + 1)],
            [1 / (38 / 121 + 1), 1 / (19 / 121 + 1), 1 / (57 / 121 + 1)],
            [1 / (57 / 121 + 1), 1 / (57 / 121 + 1), 1 / (19 / 121 + 1)],
        ]
    )
    actual = cv.co_occurrence(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_gravity():
    """Test the gravity conversion function."""
    x = np.array([[1, 2, 3], [2, 1, 3], [3, 3, 1]])
    expected = np.array(
        [
            [2.5235730726, 1.7844356324, 1.45698559277],
            [1.7844356324, 2.5235730726, 1.45698559277],
            [1.45698559277, 1.45698559277, 2.5235730726],
        ]
    )
    actual = cv.gravity(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_probability():
    """Test the probability to distance conversion function."""
    x = np.array([[0.3, 0.7], [0.5, 0.5]])
    expected = np.array([[1.8116279322, 1.1356324735], [1.3819765979, 1.3819765979]])
    actual = cv.probability(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_probability_error():
    """Test the correlation function with an out of bounds array."""
    # negative value for probability
    x = np.array([[-0.5]])
    assert_raises(ValueError, cv.probability, x)
    # too large value for probability
    y = np.array([[3]])
    assert_raises(ValueError, cv.probability, y)
    # zero value for probability (causes divide by zero issues)
    z = np.array([[0]])
    assert_raises(ValueError, cv.probability, z)


def test_covariance():
    """Test the covariance to distance conversion function."""
    x = np.array([[4, -4], [-4, 6]])
    expected = np.array([[0, 4.24264068712], [4.24264068712, 0]])
    actual = cv.covariance(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_covariance_error():
    """Test the covariance function when input contains a negative variance."""
    x = np.array([[-4, 4], [4, 6]])
    assert_raises(ValueError, cv.covariance, x)
