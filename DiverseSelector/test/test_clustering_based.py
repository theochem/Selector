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

"""Testing for the clustering-based selection algorithms."""

from DiverseSelector import clustering_based
from DiverseSelector.test.common import generate_synthetic_data
import numpy as np
import os
from numpy.testing import assert_allclose

coords, class_labels, arr_dist = generate_synthetic_data(n_samples=100,
                                                         n_features=2,
                                                         n_clusters=3,
                                                         pairwise_dist=True,
                                                         metric="euclidean",
                                                         random_state=42)


def test_kmeans():
    """Testing the kmeans selection algorithm."""
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
    """Testing the affinity propagation selection algorithm."""
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
    """Testing the mean shift selection algorithm."""
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
    """Testing the spectral selection algorithm."""
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
    """Testing the agglomerative clustering selection algorithm."""
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


def test_dbscan():
    """Testing the DBSCAN clustering selection algorithm."""
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


def test_optics():
    """Testing the OPTICS clustering selection algorithm."""
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
    """Testing the birch clustering selection algorithm."""
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


def test_gmm():
    """Testing the GMM clustering selection algorithm."""
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


def test_save_output():
    """Testing the save_output method."""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="k-means",
                                                    metric='euclidean',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster()
    selected_ids = sorted(selector.select())
    selector.save_output(selected_ids, 'test.txt', 'txt')
    selector.save_output(selected_ids, 'test.json', 'json')
    with open('test.txt', 'r') as f:
        text = f.read()
    assert text == "0\n40\n41\n47\n56\n58\n63\n67\n81\n83\n85\n86\n93\n"
    os.remove('test.txt')

    with open('test.json', 'r') as f:
        text = f.read()
    t_text = "{\"0\":40,\"1\":41,\"2\":47,\"3\":56,\"4\":58,\"5\":63," \
             "\"6\":67,\"7\":81,\"8\":83,\"9\":85,\"10\":86,\"11\":93}"
    assert text == t_text
    os.remove('test.json')


def test_subset_diversity():
    """Testing subset_diversity method from the base.py file."""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="k-means",
                                                    metric='euclidean',
                                                    arr_dist=arr_dist)
    selector.features = coords
    selector.cluster()
    selected_ids = selector.select()
    selected_diversity = selector.subset_diversity(selected_ids, metric='diversity volume')
    assert_allclose(selected_diversity, 225.60, rtol=1e-3)


def test_all_diversity():
    """Testing all_diversity property from the base.py file."""
    selector = clustering_based.ClusteringSelection(num_selected=12,
                                                    num_clusters=3,
                                                    clustering_method="k-means",
                                                    metric='euclidean',
                                                    arr_dist=arr_dist)
    selector.features = coords
    diversity = []
    for metric in ['diversity volume', 'entropy']:
        diversity.append(selector.all_diversity(metric))
    assert_allclose(diversity, [9514.79, 0], rtol=1e-3)
