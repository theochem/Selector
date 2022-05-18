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

from DiverseSelector.features import feature_reader
from DiverseSelector.metric import ComputeDistanceMatrix, entropy,\
    gini_coefficient, logdet, shannon_entropy, total_diversity_volume, wdud
from DiverseSelector.utils import PandasDataFrame
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SelectionBase(ABC):
    """Base class for subset selection."""

    def __init__(self,
                 features: Union[np.ndarray, PandasDataFrame, str, PurePath] = None,
                 arr_dist: np.array = None,
                 num_selected: int = None,
                 normalize_features: bool = False,
                 sep: str = ",",
                 engine: str = "python",
                 random_seed: int = 42,
                 **kwargs,
                 ):
        """Abstract class for other modules.

        Parameters
        ----------
        metric : str, optional
            Metric for calculating diversity of the subset "Gini", "Entropy" etc.
            Default="Tanimoto".
        random_seed : int, optional
            Random seed for reproducibility. Default=42.
        feature_type : str, optional
            Type of features. Default=None.
        mol_file : str, optional
            Path to the file with molecules. Default=None.
        feature_file : str, optional
            Path to the file with features. Default=None.
        num_selected : int, optional
            Number of molecules to select. Default=None.
        normalize_features : bool, optional
            Normalize features or not. Default=False.
        arr_dist : numpy.ndarray, optional
            Array of distances between molecules. Default=None.

        """
        self.num_selected = num_selected
        self.normalize_features = normalize_features
        self.random_seed = random_seed

        # feature loader if string is
        # accepts string or pure path object
        if features is not None:
            if isinstance(features, (str, PurePath)):
                self.features = feature_reader(file_name=features,
                                               sep=sep,
                                               engine=engine,
                                               **kwargs)
            else:
                self.features = features
            # normalize features
            if normalize_features:
                self.features = StandardScaler().fit_transform(self.features)
        # feature is None
        else:
            self.features = None

        # todo: current version only works for molecular descriptors
        # pair-wise distance matrix
        if arr_dist is None:
            dist = ComputeDistanceMatrix(feature=self.features,
                                         metric="euclidean")
            self.arr_dist = dist.compute_distance()
        else:
            self.arr_dist = arr_dist

        if self.features is None and self.arr_dist is None:
            raise ValueError("Features or distance matrix must be provided")

    # abstract method, because we want in to be in both child classes
    @abstractmethod
    def select(self):
        """Select the subset molecules with optimal diversity."""
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
