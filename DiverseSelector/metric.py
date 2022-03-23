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

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist, squareform
from DiverseSelector.utils import sklearn_supported_metrics
from sklearn.metrics import pairwise_distances


__all__ = [
    "pairwise_dist",
    "compute_diversity",
    "distance_to_similarity",
    "pairwise_similarity_bit",
    "tanimoto",
    "cosine",
    "dice",
    "bit_tanimoto",
    "bit_cosine",
    "bit_dice",
]


class ComputeDistanceMatrix:
    """Compute distance matrix.

    This class is just a demo and not finished yet.
    """

    def __init__(self,
                 feature: np.ndarray,
                 metric: str = "euclidean",
                 n_jobs: int = -1,
                 force_all_finite: bool = True,
                 **kwargs: Any,
                 ):
        """Compute pairwise distance given a feature matrix.

        Parameters
        ----------
        feature : np.ndarray
            Molecule feature matrix.
        metric : str, optional
            Distance metric.

        Returns
        -------
        dist : ndarray
            symmetric distance array.
        """
        self.feature = feature
        self.metric = metric
        self.n_jobs = n_jobs
        self.force_all_finite = force_all_finite
        self.kwargs = kwargs

    def compute_distance(self):
        """Compute the distance matrix."""
        built_in_metrics = [
            "tanimoto",
            "modified_tanimoto",
        ]
        if self.metric in sklearn_supported_metrics:
            dist = pairwise_distances(
                X=self.feature,
                Y=None,
                metric=self.metric,
                n_jobs=self.n_jobs,
                force_all_finite=self.force_all_finite,
                **self.kwargs,
            )
        elif self.metric in built_in_metrics:
            func = self._select_function(self.metric)
            dist = func(self.feature)

        return dist

    @staticmethod
    def _select_function(metric: str) -> Any:
        """Select the function to compute the distance matrix."""
        function_dict = {
            "tanimoto": tanimoto,
            "modified_tanimoto": modified_tanimoto,
        }

        return function_dict[metric]


def pairwise_dist(feature: np.array,
                  metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distance.

    Parameters
    ----------
    feature : ndarray
        feature matrix.
    metric : str
        method of calcualtion.

    Returns
    -------
    arr_dist : ndarray
        symmetric distance array.
    """
    return cdist(feature, feature, metric)


def distance_to_similarity(distance: np.array) -> np.ndarray:
    """Compute similarity.

    Parameters
    ----------
    distance : ndarray
        symmetric distance array.

    Returns
    -------
    similarity : ndarray
        symmetric similarity array.
    """
    similarity = 1 / (1 + distance)
    return similarity


def pairwise_similarity_bit(feature: np.array, metric) -> np.ndarray:
    """Compute the pairwise similarity coefficients.

    Parameters
    ----------
    feature : ndarray
        feature matrix in bit string.
    metric : str
        method of calculation.

    Returns
    -------
    pair_coeff : ndarray
        similarity coefficients for all molecule pairs in feature matrix.
    """
    pair_simi = []
    size = len(feature)
    for i in range(0, size):
        for j in range(i + 1, size):
            pair_simi.append(metric(feature[i], feature[j]))
    pair_coeff = (squareform(pair_simi) + np.identity(size))
    return pair_coeff


# this section is the similarity metrics for non-bitstring input

# todo: we need to compute the pairwise distance matrix for all the molecules in the matrix
def tanimoto(a, b) -> int:
    """Compute tanimoto coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features.
    b : array_like
        molecules B's features.

    Returns
    -------
    coeff : int
        tanimoto coefficient for molecule A and B.
    """
    coeff = (sum(a * b)) / ((sum(a ** 2)) + (sum(b ** 2)) - (sum(a * b)))
    return coeff


def cosine(a, b) -> int:
    """Compute cosine coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features.
    b : array_like
        molecules B's features.

    Returns
    -------
    coeff : int
        cosine coefficient for molecule A and B.
    """
    coeff = (sum(a * b)) / (((sum(a ** 2)) + (sum(b ** 2))) ** 0.5)
    return coeff


def dice(a, b) -> int:
    """Compute dice coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features.
    b : array_like
        molecules B's features.

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B.
    """
    coeff = (2 * (sum(a * b))) / ((sum(a ** 2)) + (sum(b ** 2)))
    return coeff


# this section is bit_string similarity calcualtions


def modified_tanimoto():
    """Compute modified tanimoto coefficient."""
    pass


def bit_tanimoto(a, b) -> int:
    """Compute tanimoto coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features in bitstring.
    b : array_like
        molecules B's features in bitstring.

    Returns
    -------
    coeff : int
        tanimoto coefficient for molecule A and B.
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for idx, _ in enumerate(a):
        if a[idx] == b[idx] and a[idx] != 0:
            c += 1
    b_t = c / (a_feat + b_feat - c)
    return b_t


def bit_cosine(a, b) -> int:
    """Compute dice coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features in bit string.
    b : array_like
        molecules B's features in bit string.

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B.
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for idx, _ in enumerate(a):
        if a[idx] == b[idx] and a[idx] != 0:
            c += 1
    b_c = c / ((a_feat * b_feat) ** 0.5)
    return b_c


def bit_dice(a, b) -> int:
    """Compute dice coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features.
    b : array_like
        molecules B's features.

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B.
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
