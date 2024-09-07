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
"""Base class for diversity based subset selection."""

import warnings
from abc import ABC, abstractmethod

import numpy as np

__all__ = ["SelectionBase"]


class SelectionBase(ABC):
    """Base class for selecting subset of sample points."""

    def select(self, x: np.ndarray, size: int, labels: np.ndarray = None) -> np.ndarray:
        """Return indices representing subset of sample points.

        Parameters
        ----------
        x: ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Feature matrix of `n_samples` samples in `n_features` dimensional feature space.
            If fun_distance is `None`, this x is treated as a square pairwise distance matrix.
        size: int
            Number of sample points to select (i.e. size of the subset).
        labels: np.ndarray, optional
            Array of integers or strings representing the labels of the clusters that
            each sample belongs to. If `None`, the samples are treated as one cluster.
            If labels are provided, selection is made from each cluster.

        Returns
        -------
        selected: list
            Indices of the selected sample points.
        """
        # check size
        if size > len(x):
            raise ValueError(
                f"Size of subset {size} cannot be larger than number of samples {len(x)}."
            )

        # if labels are not provided, indices selected from one cluster is returned
        if labels is None:
            return self.select_from_cluster(x, size)

        # check labels are consistent with number of samples
        if len(labels) != len(x):
            raise ValueError(
                f"Number of labels {len(labels)} does not match number of samples {len(x)}."
            )

        # compute the number of samples (i.e. population or pop) in each cluster
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        pop_clusters = {
            unique_label: len(np.where(labels == unique_label)[0]) for unique_label in unique_labels
        }
        # compute number of samples to be selected from each cluster
        n = size // num_clusters

        # update number of samples to select from each cluster based on the cluster population.
        # this is needed when some clusters do not have enough samples in them (pop < n) and
        # needs to be done iteratively until all remaining clusters have at least n samples
        selected_ids = []
        while np.any([value <= n for value in pop_clusters.values() if value != 0]):
            for unique_label in unique_labels:
                if pop_clusters[unique_label] != 0:
                    # get index of sample labelled with unique_label
                    cluster_ids = np.where(labels == unique_label)[0]
                    if len(cluster_ids) <= n:
                        # all samples in the cluster are selected & population becomes zero
                        selected_ids.append(cluster_ids)
                        pop_clusters[unique_label] = 0
            # update number of samples to be selected from each cluster
            totally_used_clusters = list(pop_clusters.values()).count(0)
            n = (size - len(np.hstack(selected_ids))) // (num_clusters - totally_used_clusters)

            warnings.warn(
                f"Number of molecules in one cluster is less than"
                f" {size}/{num_clusters}.\nNumber of selected "
                f"molecules might be less than desired.\nIn order to avoid this "
                f"problem. Try to use less number of clusters"
            )

        for unique_label in unique_labels:
            if pop_clusters[unique_label] != 0:
                # sample n ids from cluster labeled unique_label
                cluster_ids = np.where(labels == unique_label)[0]
                selected = self.select_from_cluster(x, n, cluster_ids)
                selected_ids.append(cluster_ids[selected])

        return np.hstack(selected_ids).flatten().tolist()

    @abstractmethod
    def select_from_cluster(
        self, x: np.ndarray, size: int, labels: np.ndarray = None
    ) -> np.ndarray:
        """Return indices representing subset of sample points from one cluster.

        Parameters
        ----------
        x: ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Feature matrix of `n_samples` samples in `n_features` dimensional feature space.
            If fun_distance is `None`, this x is treated as a square pairwise distance matrix.
        size: int
            Number of sample points to select (i.e. size of the subset).
        labels: np.ndarray, optional
            Array of integers or strings representing the labels of the clusters that
            each sample belongs to. If `None`, the samples are treated as one cluster.
            If labels are provided, selection is made from each cluster.

        Returns
        -------
        selected: list
            Indices of the selected sample points.
        """
        raise NotImplementedError
