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

from DiverseSelector.base import SelectionBase
from DiverseSelector.feature import compute_features
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
                 feature: Union[np.ndarray, PandasDataFrame, str, PurePath] = None,
                 arr_dist: np.ndarray = None,
                 normalize_features: bool = False,
                 sep: str = ",",
                 engine: str = "python",
                 clustering_method="k-means",
                 feature_file=None,
                 output=None,
                 random_seed: int = 42,
                 **kwargs
                 ):
        """Base class for clustering based subset selection."""
        #                  normalize_features: bool = False,
        #                  sep: str = ",",
        #                  engine: str = "python",
        #                  random_seed: int = 42,

        super().__init__(feature,
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
        self.feature_file = feature_file
        self.output = output
        if arr_dist is None:
            self.features = compute_features(feature_file)
        self.arr_dist = arr_dist
        self.labels = None

        if random_seed is None:
            self.random_seed = 42
        else:
            self.random_seed = random_seed

        self.__dict__.update(kwargs)

        # check if number of clusters is less than number of selected molecules
        if self.num_clusters > self.arr_dist.shape[0]:
            raise ValueError("The number of clusters is great than number of molecules.")
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
        n = self.num_selected // self.num_clusters

        selected_all = []
        for unique_label in unique_labels:
            cluster_ids = np.where(self.labels == unique_label)[0]

            np.random.seed(self.random_seed)
            selected = np.random.choice(cluster_ids, size=n, replace=False)
            selected_all.append(selected)

        selected_all = np.array(selected_all).flatten()
        return selected_all
