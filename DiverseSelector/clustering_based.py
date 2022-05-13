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

"""Clustering based compound selection."""
from pathlib import PurePath
from typing import Union
import warnings

from DiverseSelector.base import SelectionBase
from DiverseSelector.utils import PandasDataFrame
import numpy as np
from sklearn.cluster import (AffinityPropagation,
                             AgglomerativeClustering,
                             Birch,
                             DBSCAN,
                             KMeans,
                             MeanShift,
                             OPTICS,
                             SpectralClustering,
                             )
from sklearn.mixture import GaussianMixture

__all__ = [
    "ClusteringSelection",
]


class ClusteringSelection(SelectionBase):
    """Clustering based compound selection."""

    def __init__(self,
                 num_selected,
                 num_clusters,
                 features: Union[np.ndarray, PandasDataFrame, str, PurePath] = None,
                 arr_dist: np.ndarray = None,
                 normalize_features: bool = False,
                 sep: str = ",",
                 engine: str = "python",
                 clustering_method="k-means",
                 output=None,
                 random_seed: int = 42,
                 **kwargs
                 ):
        """Initializing class.

        Parameters
        ----------
        num_selected: int, optional
            Number of molecules to select. Default=None.
        num_clusters: int
            Number of clusters in the dataset.
        features: tuple
            Input feature data.
        arr_dist: np.ndarray
            2d numpy array of pairwise distances.
        normalize_features: bool, optional
            Normalize features or not. Default=False.
        sep: str
            Separating symbol in the file-to-read.
        engine: str
            Engine to use.
        clustering_method: str
            Method for performing clustering.
        output: str
            Name of the output file.
        random_seed: int
            Random seed for random sampling.
        kwargs:
            Additional arguments for performing clustering.
        """
        super().__init__(features,
                         arr_dist,
                         num_selected,
                         normalize_features,
                         sep,
                         engine,
                         random_seed)
        self.num_selected = num_selected
        self.num_clusters = num_clusters

        # the number of molecules equals the number of clusters
        self.clustering_method = clustering_method
        self.output = output
        if arr_dist is None and features is None:
            raise ValueError("Features or distance matrix must be provided")
        self.labels = None

        if random_seed is None:
            self.random_seed = 42
        else:
            self.random_seed = random_seed

        self.__dict__.update(kwargs)

        # check if we have valid number of clusters because selecting 1.5 molecules is not practical
        if int(self.num_selected / self.num_clusters) - self.num_selected / self.num_clusters != 0:
            raise ValueError("The number of molecules in each cluster should be an integer.")

        # check if selected number of clusters is less than the required number of molecules
        if self.num_clusters > self.num_selected:
            raise ValueError("The number of clusters cannot be greater than the number of "
                             "selected molecules.")

    def cluster(self, **params):
        """
        Performs clustering based on given algorithm from sklearn library and set of parameters.

        Parameters
        ----------
        params:
            set of parameters for performing clustering

        Returns
        -------
        None:
            updates the labels of the molecules

        """
        if self.clustering_method == 'k-means':
            algorithm = KMeans(n_clusters=self.num_clusters,
                               random_state=self.random_seed,
                               **params).fit(self.features)
        elif self.clustering_method == "affinity propagation":
            algorithm = AffinityPropagation(random_state=self.random_seed,
                                            **params).fit(self.features)
        elif self.clustering_method == 'mean shift':
            algorithm = MeanShift(**params).fit(self.features)
        elif self.clustering_method == 'spectral':
            algorithm = SpectralClustering(n_clusters=self.num_clusters,
                                           **params).fit(self.features)
        elif self.clustering_method == 'agglomerative':
            algorithm = AgglomerativeClustering(n_clusters=self.num_clusters,
                                                affinity=self.metric,
                                                **params).fit(self.features)
        elif self.clustering_method == 'DBSCAN':
            algorithm = DBSCAN(metric=self.metric,
                               **params).fit(self.features)
        elif self.clustering_method == 'OPTICS':
            algorithm = OPTICS(metric=self.metric,
                               **params).fit(self.features)
        elif self.clustering_method == 'birch':
            algorithm = Birch(n_clusters=self.num_clusters,
                              **params).fit(self.features)
        elif self.clustering_method == "GMM":
            labels = GaussianMixture(n_components=self.num_clusters,
                                     **params).fit_predict(self.features)
            self.labels = labels
        else:
            raise ValueError("Clustering algorithm isn't supported")

        if self.clustering_method != 'GMM':
            labels = algorithm.labels_
            self.labels = labels

    def select(self):
        """
        Selecting molecules.

        Returns
        -------
        selected_all: np.ndarray
            array of selected molecules

        """
        unique_labels = np.unique(self.labels)
        selected_all = []
        totally_used = []

        amount_molecules = np.array(
            [len(np.where(self.labels == unique_label)[0]) for unique_label in unique_labels])
        n = (self.num_selected - len(selected_all)) // (self.num_clusters - len(totally_used))
        while np.any(amount_molecules <= n):
            for unique_label in unique_labels:
                if unique_label not in totally_used:
                    cluster_ids = np.where(self.labels == unique_label)[0]
                    if len(cluster_ids) <= n:
                        selected_all.append(cluster_ids)
                        totally_used.append(unique_label)

            n = (self.num_selected - len(selected_all)) // (self.num_clusters - len(totally_used))
            amount_molecules = np.delete(amount_molecules, totally_used)

            warnings.warn(f"Number of molecules in one cluster is less than"
                          f" {self.num_selected}/{self.num_clusters}.\nNumber of selected "
                          f"molecules might be less than desired.\nIn order to avoid this "
                          f"problem. Try to use less number of clusters")

        for unique_label in unique_labels:
            if unique_label not in totally_used:
                cluster_ids = np.where(self.labels == unique_label)[0]
                np.random.seed(self.random_seed)
                selected = np.random.choice(cluster_ids, size=n, replace=False)
                selected_all.append(selected)

        selected_all = np.hstack(selected_all).flatten()
        return selected_all
