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
from DiverseSelector.metric import ComputeDistanceMatrix
from DiverseSelector.utils import PandasDataFrame
import numpy as np
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
            if isinstance(features, (str, PurePath)) and features is not None:
                self.features = feature_reader(file_name=features,
                                               sep=sep,
                                               engine=engine,
                                               **kwargs)
            # normalize features
            if normalize_features:
                self.features = StandardScaler().fit_transform(self.features)
        else:
            self.features = features

        # todo: current version only works for molecular descriptors
        # pair-wise distance matrix
        if arr_dist is None:
            dist = ComputeDistanceMatrix(feature=self.features,
                                         metric="euclidean")
            self.arr_dist = dist.compute_distance()
        else:
            self.arr_dist = arr_dist

    # abstract method, because we want in to be in both child classes
    @abstractmethod
    def select(self):
        """Select the subset molecules with optimal diversity."""
        pass

    # concrete method, because we want in to be in both child classes, and it should act
    @property
    def subset_diversity(self):
        """Calculate diversity of the subset."""
        # todo: need to implement diversity measurement here
        pass

    # concrete method, because we want in to be in both child classes, and it should act
    @property
    def all_diversity(self):
        """
        Calculate diversity of the original dataset.

        Parameters
        ----------
        div_metric:
            metric for calculating diversity of the subset ("Gini", "Entropy" etc.)

        Returns
        -------
            Scale should be discussed (0 to 1, or 0 to 100 or -1 to 1 or anything else).

        """
        # todo: need to implement diversity measurement here
        pass

    def save_output(self):
        """Save output.

        Notes
        -----
        csv or other text
        excel
        sdf

        save :
        1. index
        2. selected features
        3. selected molecules

        """
        pass
