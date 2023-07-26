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

"""Test similarity-Based Selection Methods."""

import pytest
import csv
import ast
import pkg_resources
from DiverseSelector.methods.similarity import SimilarityIndex
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

# Tests for binary data.
# ----------------------


def _get_binary_data():
    """Returns a list of binary strings.

    The proper results for the tests are known in advance.

    Returns
    -------
    list of lists
        A list of binary objects lists.
    """

    # Binary data to perform the tests, it contains 100 lists of 12 elements each
    # The proper results for the tests are known in advance.
    data = np.array(
        [
            [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
            [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
            [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        ]
    )

    return data


def _get_ref_similarity_dict():
    """Returns a dictionary with the reference values for the similarity indexes.

    The proper results for the tests are known in advance and are stored in a dictionary.

    Returns
    -------
    dict
        A dictionary with the reference values for the similarity indexes. The dictionary has the
        following structure:
            {w_factor: {c_threshold: {n_ary: value}}} where:
                w_factor: the weight factor for the similarity index
                c_threshold: the threshold value for the similarity index
    """

    # Reference values for the similarity index.
    ref_similarity_binary = {
        "fraction": {
            "None": {
                "AC": 0.1597619835634367,
                "BUB": 0.06891469248218901,
                "CT1": 0.21594387881973748,
                "CT2": 0.12273565515991067,
                "CT3": 0.09010381444416192,
                "CT4": 0.1435977860207451,
                "Fai": 0.041666666666666664,
                "Gle": 0.07428571428571429,
                "Ja": 0.078,
                "Ja0": 0.06529411764705882,
                "JT": 0.065,
                "RT": 0.05692307692307692,
                "RR": 0.021666666666666667,
                "SM": 0.06166666666666667,
                "SS1": 0.052000000000000005,
                "SS2": 0.06434782608695652,
            },
            "dissimilar": {
                "AC": 0.0,
                "BUB": 0.0,
                "CT1": 0.0,
                "CT2": 0.0,
                "CT3": 0.0,
                "CT4": 0.0,
                "Fai": 0.0,
                "Gle": 0.0,
                "Ja": 0.0,
                "Ja0": 0.0,
                "JT": 0.0,
                "RT": 0.0,
                "RR": 0.0,
                "SM": 0.0,
                "SS1": 0.0,
                "SS2": 0.0,
            },
            5: {
                "AC": 0.13861833940391668,
                "BUB": 0.043638272469757794,
                "CT1": 0.173370214880788,
                "CT2": 0.02399965239873406,
                "CT3": 0.07108193238341105,
                "CT4": 0.0791812460476248,
                "Fai": 0.03166666666666667,
                "Gle": 0.04,
                "Ja": 0.05454545454545456,
                "Ja0": 0.084,
                "JT": 0.022222222222222223,
                "RT": 0.028000000000000004,
                "RR": 0.016666666666666666,
                "SM": 0.04666666666666667,
                "SS1": 0.011764705882352941,
                "SS2": 0.07,
            },
        },
        "power_3": {
            "None": {
                "AC": 1.5127728116149686e-20,
                "BUB": 7.90100730602568e-40,
                "CT1": 0.0,
                "CT2": 0.0,
                "CT3": 0.0,
                "CT4": 0.0,
                "Fai": 5.6422575986042295e-40,
                "Gle": 1.9329988216613272e-39,
                "Ja": 2.029648762744394e-39,
                "Ja0": 5.978755024266622e-40,
                "JT": 1.6913739689536614e-39,
                "RT": 5.212247969873465e-40,
                "RR": 5.6379132298455384e-40,
                "SM": 5.646601967362921e-40,
                "SS1": 1.3530991751629292e-39,
                "SS2": 5.892106400726526e-40,
            },
            "dissimilar": {
                "AC": 0.0,
                "BUB": 0.0,
                "CT1": 0.0,
                "CT2": 0.0,
                "CT3": 0.0,
                "CT4": 0.0,
                "Fai": 0.0,
                "Gle": 0.0,
                "Ja": 0.0,
                "Ja0": 0.0,
                "JT": 0.0,
                "RT": 0.0,
                "RR": 0.0,
                "SM": 0.0,
                "SS1": 0.0,
                "SS2": 0.0,
            },
            5: {
                "AC": 1.5127727667796592e-20,
                "BUB": 6.551485125947553e-40,
                "CT1": 0.0,
                "CT2": 0.0,
                "CT3": 0.0,
                "CT4": 0.0,
                "Fai": 5.642257358488985e-40,
                "Gle": 1.3530991402370754e-39,
                "Ja": 1.8451351912323755e-39,
                "Ja0": 1.0163882938782278e-39,
                "JT": 7.517217445761529e-40,
                "RT": 3.387960979594093e-40,
                "RR": 5.637913084321148e-40,
                "SM": 5.646601632656821e-40,
                "SS1": 3.979703353638457e-40,
                "SS2": 8.469902448985232e-40,
            },
        },
    }

    return ref_similarity_binary


def _get_absolute_decimal_places_for_comparison(num1, num2, rtol=1e-4):
    """Calculate the absolute number of decimal places needed for comparison of two numbers.

    Calculate the absolute number of decimal places needed for assert_almost_equal of two numbers
    with a relative tolerance of rtol.

    Parameters
    ----------
    num1 : float
        First number.
    num2 : float
        Second number.
    rtol : float
        Relative tolerance.
    """
    max_num = max(abs(num1), abs(num2))

    # If both numbers are zero, return 5 decimal places just to avoid division by zero. Both numbers
    # are zero (and therefore equal) withing the machine precision.
    if max_num == 0:
        return 5

    # Calculate the dynamic tolerance based on the magnitudes of the numbers
    tol = rtol * max(abs(num1), abs(num2))

    return abs(int(np.log10(tol)))


# Parameter values for the tests
c_treshold_values = [None, "dissimilar", 5]
w_factor_values = ["fraction", "power_3"]
n_ary_values = [
    "AC",
    "BUB",
    "CT1",
    "CT2",
    "CT3",
    "CT4",
    "Fai",
    "Gle",
    "Ja",
    "Ja0",
    "JT",
    "RT",
    "RR",
    "SM",
    "SS1",
    "SS2",
]


@pytest.mark.parametrize("c_threshold", c_treshold_values)
@pytest.mark.parametrize("w_factor", w_factor_values)
@pytest.mark.parametrize("n_ary", n_ary_values)
def test_SimilarityIndex_call(c_threshold, w_factor, n_ary):
    """Test the similarity index for binary data.

    Test the similarity index for binary data using the reference values and several combinations
    of parameters.

    Parameters
    ----------
    c_threshold : float
        The threshold value for the similarity index.
    w_factor : float
        The weight factor for the similarity index.
    n_ary : str
        The similarity index to use.
    """

    # get the binary data
    data, ref_similarity_binary = _get_binary_data(), _get_ref_similarity_dict()

    # create instance of the class SimilarityIndex to test the similarity indexes for binary data
    si = SimilarityIndex(similarity_index=n_ary, c_threshold=c_threshold, w_factor=w_factor)

    # calculate the similarity index for the binary data
    si_value = si(data)

    # get the reference value for the similarity index
    if c_threshold == None:
        c_threshold = "None"
    ref_value = ref_similarity_binary[w_factor][c_threshold][n_ary]

    # calculate the absolute tolerance based on the relative tolerance and the magnitudes of the numbers
    tol = _get_absolute_decimal_places_for_comparison(si_value, ref_value)

    # check that the calculated value is equal to the reference value
    assert_almost_equal(si_value, ref_value, decimal=tol)
