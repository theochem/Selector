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


def compute_diversity():
    """Compute the diversity."""
    pass


def total_diversity_volume():
    """Compute the total diversity volume."""

    pass
