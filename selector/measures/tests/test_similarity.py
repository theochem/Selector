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
"""Test Similarity Module."""

import numpy as np
import pytest

from selector.measures.similarity import (
    pairwise_similarity_bit,
    tanimoto,
    modified_tanimoto,
    scaled_similarity_matrix,
    similarity_index,
)


def test_tanimoto_value():
    """Check correctness of Tanimoto similarity formula."""
    a = np.array([1, 0, 1])
    b = np.array([1, 1, 0])
    result = tanimoto(a, b)
    assert pytest.approx(result) == 1 / 3


def test_tanimoto_error():
    """Check tanimoto raises error for invalid input format."""
    with pytest.raises(ValueError):
        tanimoto(np.array([[1, 2]]), np.array([1, 2]))


def test_tanimoto_shape_mismatch():
    """Check tanimoto raises error when vector shapes differ."""
    with pytest.raises(ValueError):
        tanimoto(np.array([1, 2]), np.array([1, 2, 3]))


def test_modified_all_zero():
    """Check modified tanimoto handles zero vectors safely."""
    a = np.zeros(5)
    b = np.zeros(5)
    result = modified_tanimoto(a, b)
    assert isinstance(result, (float, np.floating))


def test_modified_partial_case():
    """Check modified tanimoto returns valid float for partial overlap."""
    a = np.array([1, 0, 1, 0])
    b = np.array([1, 1, 0, 0])
    result = modified_tanimoto(a, b)
    assert isinstance(result, (float, np.floating))


def test_invalid_metric():
    """Check pairwise_similarity_bit rejects invalid metric name."""
    X = np.array([[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        pairwise_similarity_bit(X, "wrong")


def test_pairwise_similarity_valid():
    """Check pairwise similarity returns correct matrix shape."""
    X = np.array([[1, 0], [0, 1]])
    result = pairwise_similarity_bit(X, "tanimoto")
    assert result.shape == (2, 2)


def test_pairwise_invalid_dimension():
    """Check pairwise_similarity_bit rejects 1D input."""
    X = np.array([1, 0, 1])
    with pytest.raises(ValueError):
        pairwise_similarity_bit(X, "tanimoto")


def test_scaled_identity():
    """Check scaled similarity preserves identity matrix."""
    X = np.eye(3)
    assert np.allclose(scaled_similarity_matrix(X), X)


def test_scaled_general_case():
    """Check scaled similarity preserves matrix shape."""
    X = np.array([[2.0, 1.0], [1.0, 3.0]])
    result = scaled_similarity_matrix(X)
    assert result.shape == X.shape


def test_scaled_not_square():
    """Check scaled similarity rejects non-square matrix."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        scaled_similarity_matrix(X)


def test_scaled_invalid_values():
    """Check scaled similarity rejects invalid diagonal values."""
    X = np.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError):
        scaled_similarity_matrix(X)


def test_similarity_invalid():
    """Check similarity_index rejects invalid method name."""
    x = np.array([1, 0])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        similarity_index(x, y, "bad")


def test_similarity_shape_mismatch():
    """Check similarity_index rejects mismatched vector shapes."""
    x = np.array([1, 0, 1])
    y = np.array([1, 0])
    with pytest.raises(ValueError):
        similarity_index(x, y, "RR")


def test_similarity_dimension_error():
    """Check similarity_index rejects 2D input."""
    x = np.array([[1, 0]])
    y = np.array([1, 0])
    with pytest.raises(ValueError):
        similarity_index(x, y, "RR")


@pytest.mark.parametrize(
    "idx",
    ["AC", "BUB", "CT1", "CT2", "Fai", "Gle", "Ja", "JT", "RT", "RR", "SM", "SS1", "SS2"]
)
def test_similarity_all_indices(idx):
    """Check all similarity index methods return float."""
    x = np.array([1, 0, 1])
    y = np.array([1, 1, 0])
    result = similarity_index(x, y, idx)
    assert isinstance(result, (float, np.floating))
