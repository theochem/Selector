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

"""Base class for diversity based subset selection."""

from abc import ABC, abstractmethod
import warnings

import numpy as np


class SelectionBase(ABC):
    """Base class for subset selection."""

    def select(self, arr, num_selected, labels=None):
        """
         Algorithm for selecting points.

        Parameters
        ----------
        arr: np.ndarray
            Array of features if fun_distance is provided.
            Otherwise, treated as distance matrix.
        num_selected: int
            number of points that need to be selected
        labels: np.ndarray
            labels for performing algorithm withing clusters.

        Returns
        -------
        selected: list
            list of ids of selected molecules
        """
        if labels is None:
            return self.select_from_cluster(arr, num_selected)

        # compute the number of samples (i.e. population or pop) in each cluster
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        pop_clusters = {unique_label: len(np.where(labels == unique_label)[0])
                        for unique_label in unique_labels}
        # compute number of samples to be selected from each cluster
        n = num_selected // num_clusters

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
            n = (num_selected - len(np.hstack(selected_ids))) // \
                (num_clusters - totally_used_clusters)

            warnings.warn(
                f"Number of molecules in one cluster is less than"
                f" {num_selected}/{num_clusters}.\nNumber of selected "
                f"molecules might be less than desired.\nIn order to avoid this "
                f"problem. Try to use less number of clusters"
            )

        for unique_label in unique_labels:
            if pop_clusters[unique_label] != 0:
                # sample n ids from cluster labeled unique_label
                cluster_ids = np.where(labels == unique_label)[0]
                selected = self.select_from_cluster(arr, n, cluster_ids)
                selected_ids.append(cluster_ids[selected])

        return np.hstack(selected_ids).flatten().tolist()

    @abstractmethod
    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            distance matrix for points that needs to be selected
        num_selected: int
            number of molecules that need to be selected
        cluster_ids: np.array


        Returns
        -------
        selected: list
            list of ids of molecules that are belonged to the one cluster

        """
        pass
