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

"""Testing for Utils.py."""

import DiverseSelector.utils as ut
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises


def test_sim_2_dist_float_int():
    """Test similarity to distance input handling when input is a float or int."""
    expected_1 = 0.25
    int_out = ut.sim_to_dist(4, "reciprocal")
    assert_equal(int_out, expected_1)
    expected_2 = 2
    float_out = ut.sim_to_dist(0.5, "reciprocal")
    assert_equal(float_out, expected_2)


def test_sim_2_dist_input_error():
    """Test sim to dist function with incorrect input for `x`"""
    assert_raises(ValueError, ut.sim_to_dist, np.ones([2,2,2]), "reciprocal")


def test_sim_2_dist():
    """Test similarity to distance method with specified metric"""
    x = np.array([[1, 0.2, 0.5], [0.2, 1, 0.25], [0.5, 0.25, 1]])
    expected = np.array([[0.20, 1, 0.70], [1, 0.20, 0.95], [0.70, 0.95, 0.20]])
    actual = ut.sim_to_dist(x, "reverse")
    assert_almost_equal(actual, expected, decimal=10)


def test_sim_2_dist_membership():
    """Test similarity to distance method with the membership metric."""
    x = np.array([[(1 / 2), (1 / 5)], [(1 / 4), (1 / 3)]])
    expected = np.array([[(1 / 2), (4 / 5)], [(3 / 4), (2 / 3)]])
    actual = ut.sim_to_dist(x, "membership")
    assert_almost_equal(actual, expected, decimal=10)


def test_sim_2_dist_membership_error():
    x = np.array([[1, 0, -7], [0, 1, 3], [-7, 3, 1]])
    assert_raises(ValueError, ut.sim_to_dist, x, "membership")


def test_sim_2_dist_integer():
    """Test similarity to distance method with an integer passed as the metric."""
    x = np.array([[0.5, 1], [0, 2.125]])
    expected = np.array([[2.5, 2], [3, 0.875]])
    actual = ut.sim_to_dist(x, 3)
    assert_almost_equal(actual, expected, decimal=10)


def test_reverse():
    """Test the reverse function for similarity to distance conversion."""
    x = np.array([[3, 1, 1], [1, 3, 0], [1, 0, 3]])
    expected = np.array([[0, 2, 2], [2, 0, 3], [2, 3, 0]])
    actual = ut.reverse(x)
    assert_equal(actual, expected)


def test_reciprocal():
    """Test the reverse function for similarity to distance conversion."""
    x = np.array([[1, 0.25, 0.40], [0.25, 1, 0.625], [0.40, 0.625, 1]])
    expected = np.array([[1, 4, 2.5], [4, 1, 1.6], [2.5, 1.6, 1]])
    actual = ut.reciprocal(x)
    assert_equal(actual, expected)


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
    actual = ut.exponential(x)
    assert_almost_equal(actual, expected, decimal=10)


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
    actual = ut.gaussian(x)
    assert_almost_equal(actual, expected, decimal=10)


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
    actual = ut.correlation(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_correlation_error():
    """Test the correlation function with an out of bounds array."""
    x = np.array([[1, 0, -7], [0, 1, 3], [-7, 3, 1]])
    assert_raises(ValueError, ut.correlation, x)


def test_transition():
    """Test the transition function for frequency to distance conversion."""
    x = np.array([[4, 9, 0], [9, 1, 25], [0, 25, 16]])
    root = np.array([[2, 3, 0], [3, 1, 5], [0, 5, 4]])
    expected = np.array(
        [[(1 / 2), (1 / 3), 0], [(1 / 3), 1, (1 / 5)], [0, (1 / 5), (1 / 4)]]
    )

    actual = ut.transition(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_transition_error():
    """Test the correlation function with an out of bounds array."""
    x = np.array([[1, 0, -7], [0, 1, 3], [-7, 3, 1]])
    assert_raises(ValueError, ut.correlation, x)


def test_co_occurrence():
    """Test the co-occurrence conversion function."""
    x = np.array([[1, 2, 3],
                  [2, 1, 3],
                  [3, 3, 1]])
    sum_mult_x = np.array([[19, 38, 57],
                           [38, 19, 57],
                           [57, 57, 19]])
    expected = np.array([[1/(19/121 + 1), 1/(38/121 + 1), 1/(57/121 + 1)],
                        [1/(38/121 + 1), 1/(19/121 + 1), 1/(57/121 + 1)],
                        [1/(57/121 + 1), 1/(57/121 + 1), 1/(19/121 + 1)]])
    actual = ut.co_occur(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_gravity():
    """Test the gravity conversion function."""
    x = np.array([[1, 2, 3],
                  [2, 1, 3],
                  [3, 3, 1]])
    expected = np.array([[2.5235730726, 1.7844356324, 1.45698559277],
                         [1.7844356324, 2.5235730726, 1.45698559277],
                         [1.45698559277, 1.45698559277, 2.5235730726]])
    actual = ut.gravity(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_probability():
    """Test the probability to distance conversion function."""
    x = np.array([[0.3, 0.7], [0.5, 0.5]])
    expected = np.array([[1.8116279322, 1.1356324735], [1.3819765979, 1.3819765979]])
    actual = ut.probability(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_probability_error():
    """Test the correlation function with an out of bounds array."""
    x = np.array([[1, 0, -0.5], [0, 1, 3], [-0.5, 0.2, 1]])
    assert_raises(ValueError, ut.probability, x)


def test_covariance():
    """Test the covariance to distance conversion function."""
    x = np.array([[4, -4], [-4, 6]])
    expected = np.array([[0, 4.24264068712], [4.24264068712, 0]])
    actual = ut.covariance(x)
    assert_almost_equal(actual, expected, decimal=10)


def test_dist_to_simi():
    """Testing the distance to similarity function with predefined distance matrix."""
    x = np.array([[1, 4], [3, 2]])
    actual = ut.distance_to_similarity(x, dist=True)
    expceted = np.array([[(1 / 2), (1 / 5)], [(1 / 4), (1 / 3)]])
    assert_almost_equal(actual, expceted, decimal=10)
