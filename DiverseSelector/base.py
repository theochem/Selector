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
from utils import get_features
from sklearn.preprocessing import StandardScaler


class SelectionBase(ABC):

    def __init__(self,
                 metric="Tanimoto",
                 random_seed=42,
                 feature_type=None,
                 mol_file=None,
                 feature_file=None,
                 num_selected=None):
        self.metric = metric
        self.random_seed = random_seed
        self.feature_type = feature_type
        self.mol_file = mol_file
        self.feature_file = feature_file
        self.num_selected = num_selected
        self.features = None

    @abstractmethod  # abstract method, because we want in to be in both child classes
    def select(self):
        """Select the subset molecules with optimal diversity."""
        pass

    @property
    def subset_diversity(self):  # concrete method, because we want in to be in both child classes, and it should act
        # in the same way
        """
        Calculate diversity of the subset."""
        # todo: need to implement diversity measurement here
        pass

    @property
    def all_diversity(self):  # concrete method, because we want in to be in both child classes, and it should act
        # in the same way
        """
        Calculate diversity of the original dataset.
        :param div_metric: metric for calculating diversity of the subset ("Gini", "Entropy" etc.)
        :return: float #Scale should be discussed (0 to 1, or 0 to 100 or -1 to 1 or anything else)
        """
        # todo: need to implement diversity measurement here
        pass

    def load_data(self, **kwargs):  # concrete method, because we want in to be in both child classes, and it should act
        # in the same way
        """Load dataset."""
        self.features = get_features(feature_type=self.feature_type,
                                     mol_file=self.mol_file,
                                     feature_file=self.feature_file,
                                     **kwargs)

    def save_output(self):  # concrete method, because we want in to be in both child classes, and it should act
        # in the same way
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

    def _normalize_desc(self):
        """Normalize molecular descriptors."""
        scaler = StandardScaler()
        self.features_norm = scaler.fit_transform(self.features)
        return self.features_norm
