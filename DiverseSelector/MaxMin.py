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
"""MaxMin selector."""

import numpy as np
from DiverseSelector.utils import pick_initial_compounds
import warnings


class MaxMin():
    """Selecting compounds using MinMax algorithm."""
    def __init__(self):
        """Initializing class"""
        self.arr_dist = None
        self.n_mols = None

    def select(self, arr, num_selected, func_distance=None, labels=None):
        """
         MinMax algorithm for selecting points.

        Parameters
        ----------
        arr: np.ndarray
            Array of features if fun_distance is provided.
            Otherwise, treated as distance matrix.
        func_distance: callable
            function for calculating the pairwise distance between instances of the array.
            Default is None.
        num_selected: int
            number of points that need to be selected
        labels: np.ndarray
            labels for performing algorithm withing clusters.
            Default is None.

        Returns
        -------
        selected: list
            list of ids of selected molecules
        """
        self.n_mols = arr.shape[0]
        if func_distance is not None:
            self.arr_dist = func_distance(arr)
        else:
            self.arr_dist = arr

        if labels is not None:
            unique_labels = np.unique(labels)
            num_clusters = len(unique_labels)
            selected_all = []
            totally_used = []

            amount_molecules = np.array(
                [len(np.where(labels == unique_label)[0]) for unique_label in unique_labels])

            n = (num_selected - len(selected_all)) // (num_clusters - len(totally_used))

            while np.any(amount_molecules <= n):
                for unique_label in unique_labels:
                    if unique_label not in totally_used:
                        cluster_ids = np.where(labels == unique_label)[0]
                        if len(cluster_ids) <= n:
                            selected_all.append(cluster_ids)
                            totally_used.append(unique_label)

                n = (num_selected - len(selected_all)) // (num_clusters - len(totally_used))
                amount_molecules = np.delete(amount_molecules, totally_used)

                warnings.warn(f"Number of molecules in one cluster is less than"
                              f" {num_selected}/{num_clusters}.\nNumber of selected "
                              f"molecules might be less than desired.\nIn order to avoid this "
                              f"problem. Try to use less number of clusters")

            for unique_label in unique_labels:
                if unique_label not in totally_used:
                    cluster_ids = np.where(labels == unique_label)[0]
                    selected = self.select_from_cluster(arr, n, cluster_ids)
                    selected_all.append(cluster_ids[selected])
            return np.hstack(selected_all).flatten().tolist()
        else:
            selected = self.select_from_cluster(self.arr_dist, num_selected)
            return selected

    @staticmethod
    def select_from_cluster(arr_dist, num_selected, indices=None):
        """
        MinMax algorithm for selecting points from cluster.

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
        if indices is not None:
            arr_dist = arr_dist[indices][:, indices]
        selected = [pick_initial_compounds(arr_dist)]
        while len(selected) < num_selected:
            min_distances = np.min(arr_dist[selected], axis=0)
            new_id = np.argmax(min_distances)
            selected.append(new_id)
        return selected
