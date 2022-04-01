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

"""Testing for the dissimilarity-based selection algorithms."""

import numpy as np
from DiverseSelector import clustering_based
from DiverseSelector.test.common import generate_synthetic_data

coords, class_labels, arr_dist = generate_synthetic_data(n_samples=100,
                                                    n_features=2,
                                                    n_clusters=3,
                                                    pairwise_dist=True,
                                                    metric="euclidean",
                                                    random_state=42)


def test_kmeans():
    """Testing the kmeans selection algorithm"""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="k-means",
                                                    metric='euclidian',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster()
    selected_ids = selector.select()
    labels_seleted = np.array([class_labels[elem] for elem in selected_ids])
    for i in range(3):
        assert len(labels_seleted[labels_seleted == i]) == 4


def test_affinitypropagation():
    """Testing the affinity propagation selection algorithm"""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="affinity propagation",
                                                    metric='euclidian',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster()
    selected_ids = selector.select()
    labels_seleted = np.array([class_labels[elem] for elem in selected_ids])
    for i in range(3):
        assert len(labels_seleted[labels_seleted == i]) == 4


def test_meanshift():
    """Testing the mean shift selection algorithm"""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="mean shift",
                                                    metric='euclidian',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster()
    selected_ids = selector.select()
    labels_seleted = np.array([class_labels[elem] for elem in selected_ids])
    for i in range(3):
        assert len(labels_seleted[labels_seleted == i]) == 4


def test_spectral():
    """Testing the spectral selection algorithm"""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="spectral",
                                                    metric='euclidian',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster()
    selected_ids = selector.select()
    labels_seleted = np.array([class_labels[elem] for elem in selected_ids])
    for i in range(3):
        assert len(labels_seleted[labels_seleted == i]) == 4


def test_agglomerative():
    """Testing the agglomerative clustering selection algorithm"""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="agglomerative",
                                                    metric='euclidean',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster()
    selected_ids = selector.select()
    labels_seleted = np.array([class_labels[elem] for elem in selected_ids])
    for i in range(3):
        assert len(labels_seleted[labels_seleted == i]) == 4


def test_DBSCAN():
    """Testing the DBSCAN clustering selection algorithm"""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="DBSCAN",
                                                    metric='euclidean',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster(eps=2)
    selected_ids = selector.select()
    labels_seleted = np.array([class_labels[elem] for elem in selected_ids])
    for i in range(3):
        assert len(labels_seleted[labels_seleted == i]) == 4


def test_OPTICS():
    """Testing the OPTICS clustering selection algorithm"""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="OPTICS",
                                                    metric='euclidean',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster(min_samples=15)
    selected_ids = selector.select()
    labels_seleted = np.array([class_labels[elem] for elem in selected_ids])
    for i in range(3):
        assert len(labels_seleted[labels_seleted == i]) == 4


def test_birch():
    """Testing the birch clustering selection algorithm"""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="birch",
                                                    metric='euclidean',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster()
    selected_ids = selector.select()
    labels_seleted = np.array([class_labels[elem] for elem in selected_ids])
    for i in range(3):
        assert len(labels_seleted[labels_seleted == i]) == 4


def test_GMM():
    """Testing the GMM clustering selection algorithm"""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="GMM",
                                                    metric='euclidean',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster()
    selected_ids = selector.select()
    labels_seleted = np.array([class_labels[elem] for elem in selected_ids])
    for i in range(3):
        assert len(labels_seleted[labels_seleted == i]) == 4
