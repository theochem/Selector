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


# --------------------------------------------------------------------------------------------- #
# Section of the tests for selection of indexes functions
# --------------------------------------------------------------------------------------------- #

# Many of the combinations of parameters for the tests are too stringent and the similarity of the
# samples is zero. These cases are not considered because will produce inconsistent results for the
# selection of the new indexes, and the calculation of the medoid and the outlier.

# Creating the possible combinations of the parameters for the tests. The combinations are stored
# in a list of tuples. Each tuple has the following structure: (w_factor, c_threshold, n_ary)
# where:
#   w_factor: the weight factor for the similarity index
#   c_threshold: the threshold value for the similarity index
#   n_ary: the similarity index to use


# The cases where the similarity of the samples is zero are not considered because these will be
# inconsistent with the reference values for the selection of the new indexes, and the calculation
# of the medoid and the outlier.
def _get_test_parameters():
    """Returns the parameters for the tests.

    Returns the parameters for the tests. The parameters are stored in a list of tuples. Each tuple
    has the following structure: (w_factor, c_threshold, n_ary) where:
        w_factor: the weight factor for the similarity index
        c_threshold: the threshold value for the similarity index
        n_ary: the similarity index to use

    Returns
    -------
    list
        A list of tuples with the parameters for the tests. Each tuple has the following structure:
        (w_factor, c_threshold, n_ary) where:
            w_factor: the weight factor for the similarity index
            c_threshold: the threshold value for the similarity index
            n_ary: the similarity index to use
    """
    # The cases where the similarity of the samples is zero are not considered because these will be
    # inconsistent with the reference values for the selection of the new indexes, and the calculation
    # of the medoid and the outlier.
    test_parameters = []
    for w in w_factor_values:
        for c in c_treshold_values:
            # small hack to avoid using the None value as a key in the dictionary
            c_key = c
            if c == None:
                c_key = "None"
            for n in n_ary_values:
                # ignore the cases where the similarity of the samples less than two percent
                if not _get_ref_similarity_dict()[w][c_key][n] < 0.02:
                    test_parameters.append((c, w, n))

    return test_parameters


parameters = _get_test_parameters()


def _get_ref_medoid_dict():
    """Returns a dictionary with the reference values for the medoid.

    The proper results for the tests are known in advance and are stored in a dictionary.

    The dictionary has the following structure:
        {w_factor: {c_threshold: {n_ary: value}}} where:
            w_factor: the weight factor for the similarity index
            c_threshold: the threshold value for the similarity index

    The medoid values are stored for all possible combinations of the following parameter values:
        - w_factor: None, dissimilar, 5
        - c_threshold: fraction, power_3
        - n_ary: AC, BUB, CT1, CT2, CT3, CT4, Fai, Gle, Ja, Ja0, JT, RT, RR, SM, SS1, SS2


    Returns
    -------
    dict
        A dictionary with the reference values for the medoid. The dictionary has the
        following structure:
            {w_factor: {c_threshold: {n_ary: value}}} where:
                w_factor: the weight factor for the similarity index
                c_threshold: the threshold value for the similarity index
    """
    medoid_ref_dict = {
        "None": {
            "fraction": {
                "AC": 96,
                "BUB": 69,
                "CT1": 96,
                "CT2": 80,
                "CT3": 1,
                "CT4": 23,
                "Fai": 96,
                "Gle": 69,
                "Ja": 69,
                "Ja0": 50,
                "JT": 69,
                "RT": 96,
                "RR": 1,
                "SM": 96,
                "SS1": 23,
                "SS2": 50,
            },
            "power_3": {
                "AC": 96,
                "BUB": 48,
                "CT1": 0,
                "CT2": 0,
                "CT3": 0,
                "CT4": 0,
                "Fai": 96,
                "Gle": 69,
                "Ja": 69,
                "Ja0": 50,
                "JT": 69,
                "RT": 96,
                "RR": 1,
                "SM": 96,
                "SS1": 69,
                "SS2": 50,
            },
        },
        "dissimilar": {
            "fraction": {
                "AC": 0,
                "BUB": 0,
                "CT1": 0,
                "CT2": 0,
                "CT3": 0,
                "CT4": 0,
                "Fai": 0,
                "Gle": 0,
                "Ja": 0,
                "Ja0": 0,
                "JT": 0,
                "RT": 0,
                "RR": 0,
                "SM": 0,
                "SS1": 0,
                "SS2": 0,
            },
            "power_3": {
                "AC": 0,
                "BUB": 0,
                "CT1": 0,
                "CT2": 0,
                "CT3": 0,
                "CT4": 0,
                "Fai": 0,
                "Gle": 0,
                "Ja": 0,
                "Ja0": 0,
                "JT": 0,
                "RT": 0,
                "RR": 0,
                "SM": 0,
                "SS1": 0,
                "SS2": 0,
            },
        },
        5: {
            "fraction": {
                "AC": 39,
                "BUB": 39,
                "CT1": 39,
                "CT2": 96,
                "CT3": 0,
                "CT4": 0,
                "Fai": 39,
                "Gle": 0,
                "Ja": 0,
                "Ja0": 39,
                "JT": 0,
                "RT": 39,
                "RR": 0,
                "SM": 39,
                "SS1": 0,
                "SS2": 39,
            },
            "power_3": {
                "AC": 39,
                "BUB": 39,
                "CT1": 0,
                "CT2": 0,
                "CT3": 0,
                "CT4": 0,
                "Fai": 39,
                "Gle": 0,
                "Ja": 0,
                "Ja0": 39,
                "JT": 0,
                "RT": 39,
                "RR": 0,
                "SM": 39,
                "SS1": 0,
                "SS2": 39,
            },
        },
    }

    return medoid_ref_dict


@pytest.mark.parametrize("c_threshold, w_factor, n_ary", parameters)
def test_calculate_medoid(c_threshold, w_factor, n_ary):
    """Test the function to calculate the medoid for binary data.

    Test the function to calculate the medoid for binary data using the reference values and several
    combinations of parameters. The reference values are obtained using the function
    _get_ref_medoid_dict().

    Parameters
    ----------
    c_threshold : float
        The threshold value for the similarity index.
    w_factor : float
        The weight factor for the similarity index.
    n_ary : str
        The similarity index to use.
    """

    # get the reference binary data
    data = _get_binary_data()
    # get the reference value for the medoid
    ref_medoid_dict = _get_ref_medoid_dict()

    # small hack to avoid using the None value as a key in the dictionary
    c_threshold_key = c_threshold
    if c_threshold == None:
        c_threshold_key = "None"

    ref_medoid = ref_medoid_dict[c_threshold_key][w_factor][n_ary]

    # calculate the medoid for the binary data
    medoid = SimilarityIndex().calculate_medoid(
        data, similarity_index=n_ary, w_factor=w_factor, c_threshold=c_threshold
    )

    # check that the calculated medoid is equal to the reference medoid
    assert_equal(medoid, ref_medoid)


# results from the reference code
def _get_ref_outlier_dict():
    """Returns a dictionary with the reference values for the outlier.

    The proper results for the tests are known in advance and are stored in a dictionary.

    The dictionary has the following structure:
        {w_factor: {c_threshold: {n_ary: value}}} where:
            w_factor: the weight factor for the similarity index
            c_threshold: the threshold value for the similarity index

    The outlier values are stored for all possible combinations of the following parameter values:
        - w_factor: None, dissimilar, 5
        - c_threshold: fraction, power_3
        - n_ary: AC, BUB, CT1, CT2, CT3, CT4, Fai, Gle, Ja, Ja0, JT, RT, RR, SM, SS1, SS2


    Returns
    -------
    dict
        A dictionary with the reference values for the outlier. The dictionary has the
        following structure:
            {w_factor: {c_threshold: {n_ary: value}}} where:
                w_factor: the weight factor for the similarity index
                c_threshold: the threshold value for the similarity index
    """
    oulier_ref_dict = {
        "None": {
            "fraction": {
                "AC": 4,
                "BUB": 34,
                "CT1": 4,
                "CT2": 4,
                "CT3": 8,
                "CT4": 34,
                "Fai": 34,
                "Gle": 34,
                "Ja": 34,
                "Ja0": 49,
                "JT": 34,
                "RT": 4,
                "RR": 8,
                "SM": 4,
                "SS1": 34,
                "SS2": 49,
            },
            "power_3": {
                "AC": 49,
                "BUB": 57,
                "CT1": 0,
                "CT2": 0,
                "CT3": 0,
                "CT4": 0,
                "Fai": 49,
                "Gle": 34,
                "Ja": 16,
                "Ja0": 80,
                "JT": 34,
                "RT": 4,
                "RR": 8,
                "SM": 49,
                "SS1": 34,
                "SS2": 80,
            },
        },
        "dissimilar": {
            "fraction": {
                "AC": 0,
                "BUB": 0,
                "CT1": 0,
                "CT2": 0,
                "CT3": 0,
                "CT4": 0,
                "Fai": 0,
                "Gle": 0,
                "Ja": 0,
                "Ja0": 0,
                "JT": 0,
                "RT": 0,
                "RR": 0,
                "SM": 0,
                "SS1": 0,
                "SS2": 0,
            },
            "power_3": {
                "AC": 0,
                "BUB": 0,
                "CT1": 0,
                "CT2": 0,
                "CT3": 0,
                "CT4": 0,
                "Fai": 0,
                "Gle": 0,
                "Ja": 0,
                "Ja0": 0,
                "JT": 0,
                "RT": 0,
                "RR": 0,
                "SM": 0,
                "SS1": 0,
                "SS2": 0,
            },
        },
        5: {
            "fraction": {
                "AC": 49,
                "BUB": 49,
                "CT1": 49,
                "CT2": 57,
                "CT3": 2,
                "CT4": 2,
                "Fai": 49,
                "Gle": 2,
                "Ja": 2,
                "Ja0": 49,
                "JT": 2,
                "RT": 49,
                "RR": 2,
                "SM": 49,
                "SS1": 2,
                "SS2": 49,
            },
            "power_3": {
                "AC": 49,
                "BUB": 49,
                "CT1": 0,
                "CT2": 0,
                "CT3": 0,
                "CT4": 0,
                "Fai": 49,
                "Gle": 2,
                "Ja": 2,
                "Ja0": 49,
                "JT": 2,
                "RT": 49,
                "RR": 2,
                "SM": 49,
                "SS1": 2,
                "SS2": 49,
            },
        },
    }

    return oulier_ref_dict


@pytest.mark.parametrize("c_threshold, w_factor, n_ary", parameters)
def test_calculate_outlier(c_threshold, w_factor, n_ary):
    """Test the function to calculate the outlier for binary data.

    Test the function to calculate the outlier for binary data using the reference values and several
    combinations of parameters. The reference values are obtained using the function
    _get_ref_outlier_dict().

    Parameters
    ----------
    c_threshold : float
        The threshold value for the similarity index.
    w_factor : float
        The weight factor for the similarity index.
    n_ary : str
        The similarity index to use.
    """

    # get the reference binary data
    data = _get_binary_data()
    # get the reference value for the outlier
    ref_outlier_dict = _get_ref_outlier_dict()

    # small hack to avoid using the None value as a key in the dictionary
    c_threshold_key = c_threshold
    if c_threshold == None:
        c_threshold_key = "None"

    ref_outlier = ref_outlier_dict[c_threshold_key][w_factor][n_ary]

    # calculate the outlier for the binary data
    outlier = SimilarityIndex().calculate_outlier(
        data, similarity_index=n_ary, w_factor=w_factor, c_threshold=c_threshold
    )

    # check that the calculated outlier is equal to the reference outlier
    assert_equal(outlier, ref_outlier)