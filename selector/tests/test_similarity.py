# The Selector library provides a set of tools for selecting a
# subset of the dataset and computing diversity.
#
# Copyright (C) 2023 The QC-Devs Community
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
"""Test similarity.py."""

import ast
import csv

import numpy as np
import pkg_resources
import pytest
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

from selector.methods.similarity import NSimilarity, SimilarityIndex
from selector.similarity import modified_tanimoto, pairwise_similarity_bit, tanimoto


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


def test_SimilarityIndex_init_raises():
    """Test the SimilarityIndex class for raised errors (initialization)."""
    # check raised error wrong similarity index name
    with pytest.raises(ValueError):
        SimilarityIndex(similarity_index="ttt")
    # check raised error wrong c_threshold - invalid string value
    with pytest.raises(ValueError):
        SimilarityIndex(c_threshold="ttt")
    # check raised error wrong c_threshold - invalid type (not int)
    with pytest.raises(ValueError):
        SimilarityIndex(c_threshold=1.1)


def test_SimilarityIndex_calculate_counters_raises():
    sim_idx = SimilarityIndex()

    # check raised error wrong data type
    with pytest.raises(TypeError):
        sim_idx._calculate_counters(arr=[1, 2, 3])

    # check raised error - no n_objects with data of length 1
    with pytest.raises(ValueError):
        sim_idx._calculate_counters(arr=np.array([1, 2, 3]))

    # check raised error - c_threshold bigger than n_objects
    sim_idx = SimilarityIndex(c_threshold=3)
    with pytest.raises(ValueError):
        sim_idx._calculate_counters(arr=np.array([[1, 2, 3], [4, 5, 6]]))

    # check raised error - invalid c_threshold string value
    sim_idx = SimilarityIndex()
    sim_idx.c_threshold = "ttt"
    with pytest.raises(ValueError):
        sim_idx._calculate_counters(arr=np.array([[1, 2, 3], [4, 5, 6]]))

    # check raised error - invalid weight factor string value
    sim_idx = SimilarityIndex()
    sim_idx.w_factor = "ttt"
    with pytest.raises(ValueError):
        sim_idx._calculate_counters(arr=np.array([[1, 2, 3], [4, 5, 6]]))


def test_SimilarityIndex_call_raises():
    """Test the SimilarityIndex class for raised errors (call)."""
    sim_idx = SimilarityIndex()
    # check raised error wrong data type
    with pytest.raises(TypeError):
        sim_idx(arr=[1, 2, 3])
    # check raised error - no n_objects with data of length 1
    with pytest.raises(ValueError):
        sim_idx(arr=np.array([1, 2, 3]))


def test_SimilarityIndex_calculate_medoid_raises():
    """Test the SimilarityIndex class for raised errors (calculate_medoid)."""
    sim_idx = SimilarityIndex()

    # check raised error wrong data type
    with pytest.raises(TypeError):
        sim_idx.calculate_medoid(arr=[1, 2, 3])

    # check raised error - no medoid with one dimensional data
    with pytest.raises(ValueError):
        sim_idx.calculate_medoid(arr=np.array([1, 2, 3]))

    # check raised error - no medoid with less than three samples
    with pytest.raises(ValueError):
        sim_idx.calculate_medoid(arr=np.array([[1, 2, 3], [4, 5, 6]]))

    # check raised error - c_threshold bigger than n_objects
    sim_idx = SimilarityIndex(c_threshold=4)
    with pytest.raises(ValueError):
        sim_idx.calculate_medoid(arr=np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]]))

    # check raised error - c_total and data have different number of columns
    sim_idx = SimilarityIndex()
    with pytest.raises(ValueError):
        sim_idx.calculate_medoid(
            arr=np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]]), c_total=np.array([1, 2, 3, 4])
        )


def test_SimilarityIndex_calculate_outlier_raises():
    """Test the SimilarityIndex class for raised errors (calculate_outlier)."""
    sim_idx = SimilarityIndex()

    # check raised error wrong data type
    with pytest.raises(TypeError):
        sim_idx.calculate_outlier(arr=[1, 2, 3])

    # check raised error - no medoid with one dimensional data
    with pytest.raises(ValueError):
        sim_idx.calculate_outlier(arr=np.array([1, 2, 3]))

    # check raised error - no medoid with less than three samples
    with pytest.raises(ValueError):
        sim_idx.calculate_outlier(arr=np.array([[1, 2, 3], [4, 5, 6]]))

    # check raised error - c_threshold bigger than n_objects
    sim_idx = SimilarityIndex(c_threshold=4)
    with pytest.raises(ValueError):
        sim_idx.calculate_outlier(arr=np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]]))

    # check raised error - c_total and data have different number of columns
    sim_idx = SimilarityIndex()
    with pytest.raises(ValueError):
        sim_idx.calculate_outlier(
            arr=np.array([[1, 2, 3], [4, 5, 6], [6, 7, 8]]), c_total=np.array([1, 2, 3, 4])
        )


def test_NSimilarity_init_raises():
    """Test the NSimilarity class for raised errors (initialization)."""
    # check raised error wrong similarity index name
    with pytest.raises(ValueError):
        NSimilarity(similarity_index="ttt")
    # check raised error wrong c_threshold - invalid string value
    with pytest.raises(ValueError):
        NSimilarity(c_threshold="ttt")
    # check raised error wrong c_threshold - invalid type (not int)
    with pytest.raises(ValueError):
        NSimilarity(c_threshold=1.1)


def test_NSimilarity_get_new_index_raises():
    """Test the NSimilarity class for raised errors (get_new_index)."""
    with pytest.raises(ValueError):
        NSimilarity()._get_new_index(
            arr=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            num_selected=1,
            selected_condensed=np.array([1, 2, 3]),
            select_from=np.array([1, 2, 3]),
        )


def test_NSimilarity_select_from_cluster_raises():
    """Test the NSimilarity class for raised errors (select_from_cluster)."""
    data_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    cluster_ids = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        # select more samples than the number of samples in the cluster
        NSimilarity().select_from_cluster(arr=data_array, size=5)
    with pytest.raises(ValueError):
        # start from sample index that is not in the cluster
        NSimilarity().select_from_cluster(arr=data_array, size=2, start=[4])
    with pytest.raises(ValueError):
        # start from sample index that is not in the cluster
        NSimilarity().select_from_cluster(arr=data_array, size=2, start=[4])
    with pytest.raises(ValueError):
        # start from invalid string value of start
        NSimilarity().select_from_cluster(arr=data_array, size=2, start="ttt")
    with pytest.raises(ValueError):
        # start from invalid type of start
        NSimilarity().select_from_cluster(arr=data_array, size=2, start=[1.2])
    with pytest.raises(ValueError):
        # try to data not scaled between 0 and 1
        NSimilarity(preprocess_data=False).select_from_cluster(arr=data_array, size=2)
    with pytest.raises(ValueError):
        # try to use starting index that is not in the cluster
        NSimilarity().select_from_cluster(
            arr=data_array, size=2, start=[4], cluster_ids=cluster_ids
        )
    with pytest.raises(ValueError):
        # try to use invalid starting index
        NSimilarity().select_from_cluster(
            arr=data_array, size=2, start=4.2, cluster_ids=cluster_ids
        )


# --------------------------------------------------------------------------------------------- #
# Tests for the function results of the SimilarityIndex and NSimilarity classes.
# --------------------------------------------------------------------------------------------- #
# The following part tests the results of the SimilarityIndex and NSimilarity classes methods
# for a set of binary data. The proper results for the tests are known in advance.

# Tests for binary data.
# --------------------------------------------------------------------------------------------- #

# --------------------------------------------------------------------------------------------- #
# Section of the test for the SimilarityIndex class
# --------------------------------------------------------------------------------------------- #


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
    sim_idx = SimilarityIndex(similarity_index=n_ary, c_threshold=c_threshold, w_factor=w_factor)

    # calculate the similarity index for the binary data
    sim_idx_value = sim_idx(data)

    # get the reference value for the similarity index
    if c_threshold is None:
        c_threshold = "None"
    ref_value = ref_similarity_binary[w_factor][c_threshold][n_ary]

    # calculate the absolute tolerance based on the relative tolerance and the numbers magnitude
    tol = _get_absolute_decimal_places_for_comparison(sim_idx_value, ref_value)

    # check that the calculated value is equal to the reference value
    assert_almost_equal(sim_idx_value, ref_value, decimal=tol)


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
    # inconsistent with the reference values for the selection of the new indexes, and the
    # calculation of the medoid and the outlier.
    test_parameters = []
    for w in w_factor_values:
        for c in c_treshold_values:
            # small hack to avoid using the None value as a key in the dictionary
            c_key = c
            if c is None:
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
    if c_threshold is None:
        c_threshold_key = "None"

    ref_medoid = ref_medoid_dict[c_threshold_key][w_factor][n_ary]

    # calculate the medoid for the binary data
    sim_idx = SimilarityIndex(similarity_index=n_ary, c_threshold=c_threshold, w_factor=w_factor)
    medoid = sim_idx.calculate_medoid(arr=data)

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

    Test the function to calculate the outlier for binary data using the reference values and
    several combinations of parameters. The reference values are obtained using the function
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
    if c_threshold is None:
        c_threshold_key = "None"

    ref_outlier = ref_outlier_dict[c_threshold_key][w_factor][n_ary]

    # calculate the outlier for the binary data
    sim_idx = SimilarityIndex(similarity_index=n_ary, c_threshold=c_threshold, w_factor=w_factor)
    outlier = sim_idx.calculate_outlier(arr=data)

    # check that the calculated outlier is equal to the reference outlier
    assert_equal(outlier, ref_outlier)


# --------------------------------------------------------------------------------------------- #
# Section of the test for the NSimilarity class
# --------------------------------------------------------------------------------------------- #


def _get_ref_new_index():
    """
    Returns reference data for testing the function to calculate the new index.

    Returns the tuple (selected_points, new_indexes_dict) where selected_points is a list of
    selected samples given to the function (get_new_index) and new_indexes_dict is  a dictionary
    with the reference values for the new index to be selected based on the previously selected
    indices and the other function parameters.

    Returns
    -------
    tuple (list, dict)
        A tuple with the reference data for testing the function to calculate the new index. The
        tuple has the following structure:
            (selected_points, new_indexes_dict) where:
                selected_points: a list of selected samples given to the function (get_new_index)
                new_indexes_dict: a dictionary with the reference values for the new index to be
                    selected based on the previously selected indices and the other function
                    parameters. The dictionary has the following structure:
                        {w_factor: {c_threshold: {n_ary: value}}} where:
                            w_factor: the weight factor for the similarity index
                            c_threshold: the threshold value for the similarity index
                            n_ary: the similarity index to use

    """

    # The selected points to be given to the function (get_new_index)
    selected_samples = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    # The reference values for the new index to be selected based on the previously selected
    # indices and the other function parameters.
    new_index_dict = {
        "None": {
            "fraction": {
                "AC": 62,
                "BUB": 62,
                "CT1": 62,
                "CT2": 62,
                "CT3": 2,
                "CT4": 44,
                "Fai": 62,
                "Gle": 44,
                "Ja": 44,
                "Ja0": 62,
                "JT": 44,
                "RT": 62,
                "RR": 2,
                "SM": 62,
                "SS1": 44,
                "SS2": 62,
            },
            "power_3": {
                "AC": 67,
                "BUB": 11,
                "CT1": 67,
                "CT2": 20,
                "CT3": 2,
                "CT4": 44,
                "Fai": 11,
                "Gle": 44,
                "Ja": 44,
                "Ja0": 11,
                "JT": 44,
                "RT": 67,
                "RR": 2,
                "SM": 67,
                "SS1": 44,
                "SS2": 11,
            },
        },
        5: {
            "fraction": {
                "AC": 2,
                "BUB": 2,
                "CT1": 2,
                "CT2": 2,
                "CT3": 2,
                "CT4": 2,
                "Fai": 2,
                "Gle": 2,
                "Ja": 2,
                "Ja0": 2,
                "JT": 2,
                "RT": 2,
                "RR": 2,
                "SM": 2,
                "SS1": 2,
                "SS2": 2,
            },
            "power_3": {
                "AC": 2,
                "BUB": 2,
                "CT1": 2,
                "CT2": 2,
                "CT3": 2,
                "CT4": 2,
                "Fai": 2,
                "Gle": 2,
                "Ja": 2,
                "Ja0": 2,
                "JT": 2,
                "RT": 2,
                "RR": 2,
                "SM": 2,
                "SS1": 2,
                "SS2": 2,
            },
        },
    }

    return selected_samples, new_index_dict


@pytest.mark.parametrize("c_threshold, w_factor, n_ary", parameters)
def test_get_new_index(c_threshold, w_factor, n_ary):
    """Test the function get a new sample from the binary data.

    Test the function to get a new sample from the binary data using the reference values and
    several combinations of parameters. The reference values are obtained using the function
    _get_ref_new_index().

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
    selected_samples, ref_new_index_dict = _get_ref_new_index()

    # columnwise sum of the selected samples
    selected_condensed_data = np.sum(np.take(data, selected_samples, axis=0), axis=0)
    # indices of the samples to select from
    select_from_n = [i for i in range(len(data)) if i not in selected_samples]
    # number of samples that are already selected
    n = len(selected_samples)

    # small hack to avoid using the None value as a key in the dictionary
    c_threshold_key = c_threshold
    if c_threshold is None:
        c_threshold_key = "None"

    ref_new_index = ref_new_index_dict[c_threshold_key][w_factor][n_ary]

    # calculate the outlier for the binary data
    nsi = NSimilarity(similarity_index=n_ary, w_factor=w_factor, c_threshold=c_threshold)

    new_index = nsi._get_new_index(
        arr=data,
        selected_condensed=selected_condensed_data,
        select_from=select_from_n,
        num_selected=n,
    )

    assert_equal(new_index, ref_new_index)


# --------------------------------------------------------------------------------------------- #
# Selecting a subset of the parameters generated for the tests
# --------------------------------------------------------------------------------------------- #

# remove cases where c_threshold is 5. This is not a valid case for the selection of the next point
# if the number of selected samples is less than 5
parameters = [x for x in parameters if not x[0] == 5]

# The start parameter can be a list of elements that need to be selected first.
# The case of the three elements included in a selection  1, 2, 3 is tested.
start_values = ["medoid", "outlier", [1, 2, 3]]
# sample size values to test
sample_size_values = [10, 20, 30]

# --------------------------------------------------------------------------------------------- #
# Get reference data for testing the selection of the diverse subset
# --------------------------------------------------------------------------------------------- #


def get_data_file_path(file_name):
    """Get the absolute path of the data file inside the package.

    Parameters
    ----------
    file_name : str
        The name of the data file to load.

    Returns
    -------
    str
        The absolute path of the data file inside the package
    """
    data_file_path = pkg_resources.resource_filename(__name__, f"data/{file_name}")
    return data_file_path


def _get_selections_ref_dict():
    """Returns a dictionary with the reference values for the selection of samples.

    The proper results for the tests are known in advance and are stored in a csv file.
    The file is read and the values are stored in a dictionary. The dictionary has the following
    structure:
        {w_factor: {c_threshold: {n_ary: {sample_size: {start: value}}}}} where:
            w_factor: the weight factor for the similarity index
            c_threshold: the threshold value for the similarity index
            n_ary: the similarity index to use
            sample_size: the number of samples to select
            start: the method to use to select the first(s) sample(s)
    """

    file_path = get_data_file_path("ref_similarity_data.csv")
    with open(file_path, encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        next(reader)  # skip header
        # initialize the dictionary
        data_dict = {}

        for row in reader:
            # The first column is the c_threshold
            data_dict[row[0]] = data_dict.get(row[0], {})
            # The second column is the w_factor
            data_dict[row[0]][row[1]] = data_dict[row[0]].get(row[1], {})
            # The third column is the sample_size
            data_dict[row[0]][row[1]][row[2]] = data_dict[row[0]][row[1]].get(row[2], {})
            # The fourth column is the start_idx
            data_dict[row[0]][row[1]][row[2]][row[3]] = data_dict[row[0]][row[1]][row[2]].get(
                row[3], {}
            )
            # The fifth column stores the n_ary and the sixth column stores the reference value
            data_dict[row[0]][row[1]][row[2]][row[3]][row[4]] = ast.literal_eval(row[5])
    return data_dict


print(_get_selections_ref_dict())


@pytest.mark.parametrize("c_threshold, w_factor, n_ary", parameters)
@pytest.mark.parametrize("sample_size", sample_size_values)
@pytest.mark.parametrize("start", start_values)
def test_NSimilarity_select(c_threshold, w_factor, sample_size, n_ary, start):
    """
    Test the diversity selection methods based on similarity indexes for binary data.

    Parameters
    ----------
    c_threshold : float
        The threshold value for the similarity index.
    w_factor : float
        The weight factor for the similarity index.
    sample_size : int
        The number of molecules to select.
    n_ary : str
        The similarity index to use.
    start : str
        The method to use to select the first(s) sample(s).

    """

    # get the binary data
    data = _get_binary_data()
    # get the reference selected data
    reference_selected_data = _get_selections_ref_dict()

    # create instance of the class SimilarityIndex to test the similarity indexes for binary data
    selector = NSimilarity(
        similarity_index=n_ary,
        w_factor=w_factor,
        c_threshold=c_threshold,
        preprocess_data=False,
    )
    # select the diverse subset using the similarity index
    selected_data = selector.select_from_cluster(data, size=sample_size, start=start)

    # get the reference value for the similarity index
    # transform invalid keys to strings
    if c_threshold is None:
        c_threshold = "None"
    # transform sample_size to string (as it is used as a key in the dictionary)
    sample_size = str(sample_size)
    # lists of initial indexes are not valid keys in the dictionary (non hashable)
    if isinstance(start, list):
        start = str(start).replace(" ", "")
    # get the reference list as a string
    ref_list = reference_selected_data[c_threshold][w_factor][sample_size][start][n_ary]

    # check if the selected data has the same size as the reference data
    assert_equal(len(selected_data), len(ref_list))

    # check if the selected data is equal to the reference data
    assert all(x in ref_list for x in selected_data)
