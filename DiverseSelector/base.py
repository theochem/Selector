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

from DiverseSelector.feature import get_features
from sklearn.preprocessing import StandardScaler


class SelectionBase(ABC):
    """Base class for subset selection."""

    def __init__(self,
                 metric: str = "Tanimoto",
                 random_seed: int = 42,
                 feature_type: str = None,
                 mol_file: str = None,
                 feature_file: str = None,
                 num_selected: str = None,
                 normalize_features: bool = False,
                 arr_dist=None,
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
        self.metric = metric
        self.random_seed = random_seed
        self.feature_type = feature_type
        self.mol_file = mol_file
        self.feature_file = feature_file
        self.num_selected = num_selected
        self.normalize_features = normalize_features
        self.arr_dist = arr_dist
        if arr_dist is not None:
            self.features = self.load_data(**kwargs)

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

    def load_data(self, **kwargs):
        """Load dataset."""
        self.features = get_features(feature_type=self.feature_type,
                                     mol_file=self.mol_file,
                                     feature_file=self.feature_file,
                                     **kwargs)
        # normalize the features when needed
        if self.normalize_features:
            self.features = StandardScaler().fit_transform(self.features)

        return self.features

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
