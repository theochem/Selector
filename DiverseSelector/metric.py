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

"""Metric calculation module."""

import numpy as np
from scipy.spatial.distance import cdist, squareform

__all__ = [
    "pairwise_dist",
    "compute_diversity",
    "distance_similarity",
    "pairwise_similarity",
    "pairwise_similarity_bit",
    "tanimoto",
    "cosine",
    "dice",
    "bit_tanimoto",
    "bit_cosine",
    "bit_dice",
]


def pairwise_dist(feature: np.array,
                  metric: str = "euclidean"):
    """Compute pairwise distance.

    Parameters
    ----------
    feature : ndarray
        feature matrix
    metric : str
        metric to use

    Returns
    -------
    arr_dist : ndarray
        symmetric distance array
    """
    if metric == "euclidean":
        arr_dist = cdist(feature, feature, "euclidean")
    else:
        arr_dist = cdist(feature, feature, metric)
    return arr_dist


def distance_similarity(distance: np.array):
    """Compute similarity.

    Parameters
    ----------
    distance : ndarray
        symmetric distance array

    Returns
    -------
    similarity : ndarray
        symmetric similarity array.
    """
    similarity = 1 / (1 + distance)
    return similarity


def pairwise_similarity(feature: np.array, metric):
    """Compute the pairwaise similarity coefficients.

    Parameters
    ----------
    feature : ndarray
        feature matrix
    metric : str
        method of calculation

    Returns
    -------
    pair_coeff : ndarray
        similairty coefficients for all molecule pairs in feature matrix.
    """
    pair_simi = []
    size = len(np.shape(feature))
    for i in range(0, size + 1):
        for j in range(i + 1, size + 1):
            pair_simi.append((metric(feature[:, i], feature[:, j])))
    # this only works when the similarity to self is equal to 1
    pair_coeff = (squareform(pair_simi) + np.identity(size + 1))
    return pair_coeff


def pairwise_similarity_bit(feature: np.array, metric):
    """Compute the pairwise similarity coefficients.

    Parameters
    ----------
    feature : ndarray
        feature matrix in bit string
    metric : str
        method of calculation

    Returns
    -------
    pair_coeff : ndarray
        similarity coefficients for all molecule pairs in feature matrix
    """
    pair_simi = []
    size = len(feature)
    for i in range(0, size):
        for j in range(i + 1, size):
            pair_simi.append(metric(feature[i], feature[j]))
    pair_coeff = (squareform(pair_simi) + np.identity(size))
    return pair_coeff


# this section is the similarity metrics for non-bitstring input


def tanimoto(a, b):
    """Compute tanimoto coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features
    b : array_like
        molecules B's features

    Returns
    -------
    coeff : int
        tanimoto coefficient for molecule A and B
    """
    coeff = (sum(a * b)) / ((sum(a ** 2)) + (sum(b ** 2)) - (sum(a * b)))
    return coeff


def cosine(a, b):
    """Compute cosine coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features
    b : array_like
        molecules B's features

    Returns
    -------
    coeff : int
        cosine coefficient for molecule A and B
    """
    coeff = (sum(a * b)) / (((sum(a ** 2)) + (sum(b ** 2))) ** 0.5)
    return coeff


def dice(a, b):
    """Compute dice coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features
    b : array_like
        molecules B's features

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B
    """
    coeff = (2 * (sum(a * b))) / ((sum(a ** 2)) + (sum(b ** 2)))
    return coeff


# this section is bit_string similarity calcualtions


def modified_tanimoto():
    """Compute modified tanimoto coefficient."""
    pass


def bit_tanimoto(a, b):
    """Compute tanimoto coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features in bitstring
    b : array_like
        molecules B's features in bitstring

    Returns
    -------
    coeff : int
        tanimoto coefficient for molecule A and B
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for idx, _ in enumerate(a):
        if a[idx] == b[idx] and a[idx] != 0:
            c += 1
    b_t = c / (a_feat + b_feat - c)
    return b_t


def bit_cosine(a, b):
    """Compute dice coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features in bit string
    b : array_like
        molecules B's features in bit string

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for idx, _ in enumerate(a):
        if a[idx] == b[idx] and a[idx] != 0:
            c += 1
    b_c = c / ((a_feat * b_feat) ** 0.5)
    return b_c


def bit_dice(a, b):
    """Compute dice coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features
    b : array_like
        molecules B's features

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for idx, _ in enumerate(a):
        if a[idx] == b[idx] and a[idx] != 0:
            c += 1
    b_d = (2 * c) / (a_feat + b_feat)
    return b_d


def compute_diversity():
    """Compute the diversity."""
    pass


def total_diversity_volume():
    """Compute the total diversity volume."""
    pass
