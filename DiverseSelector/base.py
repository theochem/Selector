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
import numpy as np
import warnings


class SelectionBase(ABC):
    """Base class for subset selection."""

    def select(self, arr, num_selected, labels=None):
        """
         MinMax algorithm for selecting points.

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

        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)
        selected_all = []
        totally_used = []

        amount_molecules = np.array(
            [
                len(np.where(labels == unique_label)[0])
                for unique_label in unique_labels
            ]
        )

        n = (num_selected - len(selected_all)) // (num_clusters - len(totally_used))

        while np.any(amount_molecules <= n):
            for unique_label in unique_labels:
                if unique_label not in totally_used:
                    cluster_ids = np.where(labels == unique_label)[0]
                    if len(cluster_ids) <= n:
                        selected_all.append(cluster_ids)
                        totally_used.append(unique_label)

            n = (num_selected - len(selected_all)) // (
                num_clusters - len(totally_used)
            )
            amount_molecules = np.delete(amount_molecules, totally_used)

            warnings.warn(
                f"Number of molecules in one cluster is less than"
                f" {num_selected}/{num_clusters}.\nNumber of selected "
                f"molecules might be less than desired.\nIn order to avoid this "
                f"problem. Try to use less number of clusters"
            )

        for unique_label in unique_labels:
            if unique_label not in totally_used:
                cluster_ids = np.where(labels == unique_label)[0]
                selected = self.select_from_cluster(arr, n, cluster_ids)
                selected_all.append(cluster_ids[selected])

        return np.hstack(selected_all).flatten().tolist()

    @staticmethod
    @abstractmethod
    def select_from_cluster(arr_dist, num_selected):
        """
        Algorithm for selecting points from cluster.

        Parameters
        ----------
        arr_dist: np.ndarray
            distance matrix for points that needs to be selected
        num_selected: int
            number of molecules that need to be selected

        Returns
        -------
        selected: list
            list of ids of selected molecules

        """
        pass
