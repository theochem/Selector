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
from pathlib import PurePath
from typing import Union

from DiverseSelector.feature import feature_reader
from DiverseSelector.metric import compute_distance_matrix, entropy,\
    gini_coefficient, logdet, shannon_entropy, total_diversity_volume, wdud
from DiverseSelector.utils import PandasDataFrame
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings


class SelectionBase(ABC):
    """Base class for subset selection."""

    def __init__(self):
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

    # concrete method, because we want in to be in both child classes, and it should act
    def subset_diversity(self, indices, metric):
        """
        Calculate diversity of the subset.

        Parameters
        ----------
        indices: np.ndarray
            indices of the subset diversity of which one wants to calculate.
        metric: str
            metric for calculating diversity. Default is 'diversity volume'.
            Other options are 'entropy', 'diversity index', 'logdet', 'shannon entropy', 'wdud'.

        Returns
        -------
        score: float
            diversity volume

        Notes
        -----
        Agrafiotis, D. K.. (1997) Stochastic Algorithms for Maximizing Molecular Diversity.
        Journal of Chemical Information and Computer Sciences 37, 841-851.
        """
        if isinstance(self.features, np.ndarray):
            mtrx = self.features[indices]
        elif isinstance(self.features, pd.DataFrame):
            mtrx = self.features.iloc[indices, :].to_numpy()
        else:
            raise ValueError("features should be a numpy.ndarray or pandas.DataFrame object")

        if metric == 'diversity volume':
            score = total_diversity_volume(mtrx)
        elif metric == 'entropy':
            score = entropy(mtrx)
        elif metric == 'logdet':
            score = logdet(mtrx)
        elif metric == 'shannon entropy':
            score = shannon_entropy(mtrx)
        elif metric == 'wdud':
            score = wdud(mtrx)
        return score

    # concrete method, because we want in to be in both child classes, and it should act
    def all_diversity(self, metric='diversity volume'):
        """
        Calculate diversity of the original dataset.

        Returns
        -------
        score: float
            diversity volume.
        metric: str
            metric for calculating diversity. Default is 'diversity volume'.
            Other options are 'entropy', 'diversity index', 'logdet',
             'shannon entropy', 'wdud', 'gini'.

        Notes
        -----
        All methods have references in the metric.py file
        """
        if isinstance(self.features, np.ndarray):
            mtrx = self.features
        elif isinstance(self.features, pd.DataFrame):
            mtrx = self.features.to_numpy()
        else:
            raise ValueError("features should be a numpy.ndarray or pandas.DataFrame object")

        if metric == 'diversity volume':
            score = total_diversity_volume(mtrx)
        elif metric == 'entropy':
            score = entropy(mtrx)
        elif metric == 'logdet':
            score = logdet(mtrx)
        elif metric == 'shannon entropy':
            score = shannon_entropy(mtrx)
        elif metric == 'wdud':
            score = wdud(mtrx)
        elif metric == 'gini':
            score = gini_coefficient(mtrx)

        return score

    @staticmethod
    def save_output(selected, fname, frmt='txt', sep=' ', **kwargs):
        """
        Save the selected ids of molecules to file.

        Parameters
        ----------
        selected: np.ndarray
            numpy array of selected molecules.
        fname: str
            filename to save output; must include the extension.
        format: str
            'txt', 'json', 'csv', 'excel' file format of the output file.
            If 'txt' format is chosen, then sep can be specified is a separation character.
        sep: str
            separator between lines.
        kwargs: dict
            other arguments for supporting the json and excel file formats,
            that are accepted by pandas.

        Returns
        -------
        None.
        """
        series = pd.Series(selected)
        if frmt in ('txt', 'csv'):
            series.to_csv(fname, sep, index=False)
        elif frmt == 'json':
            series.to_json(fname, **kwargs)
        elif frmt == 'excel':
            series.to_excel(fname, index=False, **kwargs)
        else:
            raise ValueError("Wrong file format")
