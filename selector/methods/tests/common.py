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
"""Common functions for test module."""

from typing import Any, Tuple, Union

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

__all__ = [
    "generate_synthetic_data",
]


def generate_synthetic_data(
    n_samples: int = 100,
    n_features: int = 2,
    n_clusters: int = 2,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    metric: str = "euclidean",
    shuffle: bool = True,
    random_state: int = 42,
    pairwise_dist: bool = False,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate synthetic data.

    Parameters
    ----------
    n_samples : int, optional
        The number of sample points.
    n_features : int, optional
        The number of features.
    n_clusters : int, optional
        The number of clusters.
    cluster_std : float, optional
        The standard deviation of the clusters.
    center_box : tuple[float, float], optional
        The bounding box for each cluster center when centers are generated at random.
    metric : str, optional
        The metric used for computing pairwise distances. For the supported
        distance matrix, please refer to
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html.
    shuffle : bool, optional
        Whether to shuffle the samples.
    random_state : int, optional
        The random state used for generating synthetic data.
    pairwise_dist : bool, optional
        If True, then compute and return the pairwise distances between sample points.
    **kwargs : Any, optional
            Additional keyword arguments for the scikit-learn `pairwise_distances` function.

    Returns
    -------
    syn_data : np.ndarray
        The synthetic data.
    class_labels : np.ndarray
        The integer labels for cluster membership of each sample.
    dist: np.ndarray
        The symmetric pairwise distances between samples.

    """
    # pylint: disable=W0632
    syn_data, class_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        center_box=center_box,
        shuffle=shuffle,
        random_state=random_state,
        return_centers=False,
    )
    if pairwise_dist:
        dist = pairwise_distances(
            X=syn_data,
            Y=None,
            metric=metric,
            **kwargs,
        )
        return syn_data, class_labels, dist
    return syn_data, class_labels
