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
from sklearn.metrics import pairwise_distances
from typing import Any

__all__ = [
    "pairwise_dist",
    "compute_diversity",
    "distance_to_similarity",
    "pairwise_similarity",
    "pairwise_similarity_bit",
    "tanimoto",
    "cosine",
    "dice",
    "bit_tanimoto",
    "bit_cosine",
    "bit_dice",
]

sklearn_supported_metrics = ["cityblock",
                              "cosine",
                              "euclidean",
                              "l1",
                              "l2",
                              "manhattan",
                              "braycurtis",
                              "canberra",
                              "chebyshev",
                              "correlation",
                              "dice",
                              "hamming",
                              "jaccard",
                              "kulsinski",
                              "mahalanobis",
                              "minkowski",
                              "rogerstanimoto",
                              "russellrao",
                              "seuclidean",
                              "sokalmichener",
                              "sokalsneath",
                              "sqeuclidean",
                              "yule",
                              ]


class ComputeDistanceMatrix:
    """Compute distance matrix.

    This class is just a demo and not finished yet."""

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
        print(dist)
        return dist

    @staticmethod
    def _select_function(metric: str) -> Any:
        """Select the function to compute the distance matrix."""
        function_dict = {
            "tanimoto": tanimoto,
            "modified_tanimoto": modified_tanimoto,
        }

        return function_dict[metric]


def distance_similarity(x, dist = True):
    """Convert between distance and similarity matrix.
    
    Parameters
    ----------
    distance : ndarray
        symmetric distance array.

    Returns
    -------
    similarity : ndarray
        symmetric similarity array.
    """
    if dist == True:
        y = 1 / (1 + x)
    else:
        y = (1 / x) - 1
    return y


def pairwise_similarity_bit(feature: np.array, metric):
    """Compute the pairwaise similarity coefficients.
    
    Parameters
    ----------
    feature : ndarray
        feature matrix.
    metric : str
        method of calculation.

    Returns
    -------
    pair_coeff : ndarray
        similairty coefficients for all molecule pairs in feature matrix.
    """
    pair_simi = []
    size = len(feature)
    for i in range(0, size):
        for j in range(i + 1 , size):
            pair_simi.append(metric(feature[i],feature[j]))
    pair_coeff = (squareform(pair_simi) + np.identity(size))
    return pair_coeff


def tanimoto(a, b):
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


def bit_tanimoto(a ,b):
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


def modified_tanimoto(a, b):
    # This is not finished
    n = len(a)
    n_11 = sum(a * b)
    n_00 = sum((1 - a) * (1 - b))
    if n_00 == n:
        t_1 = 1
    else:
        t_1 = n_11 / (n_00 - n)
    if n_11 == n:
        t_0 = 1
    else:
        t_0 = n_00 / (n - n_11)
    p = ((n - n_00) + n_11) / (2 * n)
    mt = (((2 - p) / 3) * t_1) + (((1 + p) / 3) * t_0)
    return mt

