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

from .utils import get_features


class SelectionBase(ABC):

    @abstractmethod
    def select(self):
        """Select the subset molecules with optimal diversity."""
        pass

    @abstractmethod
    @property
    def subset_diversity(self):
        """
        Calculate diversity of the subset."""
        # todo: need to implement diversity measurement here
        pass

    @abstractmethod
    @property
    def all_diversity(self):
        """
        Calculate diversity of the original dataset.
        :param div_metric: metric for calculating diversity of the subset ("Gini", "Entropy" etc.)
        :return: float #Scale should be discussed (0 to 1, or 0 to 100 or -1 to 1 or anything else)
        """
        # todo: need to implement diversity measurement here
        pass

    @abstractmethod
    def load_data(self):
        """Load dataset."""
        pass

    @abstractmethod
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


class DissimilaritySelectionBase(SelectionBase):
    def __init__(self,
                 initialization="medoid",
                 metric="Tanimoto",
                 random_seed=42,
                 feature_type=None,
                 mol_file=None,
                 feature_file=None,
                 num_selected=None,
                 **kwargs,
                 ):
        """Base class for dissimilarity based subset selection."""
        self.initialization = initialization
        self.random_seed = random_seed
        self.metric = metric
        self.num_selected = num_selected

        # compute/load molecular features
        self.features = get_features(feature_type=feature_type,
                                     mol_file=mol_file,
                                     feature_file=feature_file,
                                     **kwargs)

    def pick_initial_compounds(self):
        """Pick the initial compounds."""
        pass

    def compute_diversity(self):
        """Compute the distance metrics."""
        # for iterative selection and final subset both
        pass

    def select(self):
        """
        Select the subset molecules with optimal diversity.
        :param num_selected: amount of molecule that need to be selected
        :param random_seed: int
        :return:
        """
        if self.random_seed is None:
            self.random_seed = 42

        pass

    @property
    def subset_diversity(self):
        """
        Calculate diversity of the subset."""
        # todo: need to implement diversity measurement here
        pass

    @property
    def all_diversity(self):
        """
        Calculate diversity of the original dataset.
        :param div_metric: metric for calculating diversity of the subset ("Gini", "Entropy" etc.)
        :return: float #Scale should be discussed (0 to 1, or 0 to 100 or -1 to 1 or anything else)
        """
        # todo: need to implement diversity measurement here
        pass

    def load_data(self):
        """Load dataset."""
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


class ClusteringSelectionBase(SelectionBase):
    def __init__(self,
                 num_selected,
                 num_clusters,
                 clustering_method="k-means",
                 metric="Tanimoto",
                 feature_file=None,
                 output=None,
                 random_seed=None,
                 **kwargs
                 ):
        """Base class for clustering based subset selection."""

        self.num_selected = num_selected
        self.num_clusters = num_clusters

        # the number of molecules equals the number of clusters
        self.clustering_method = clustering_method
        self.metric = metric
        self.feature_file = feature_file
        self.output = output

        if random_seed is None:
            self.random_seed = 42
        else:
            self.random_seed = random_seed

        self.__dict__.update(kwargs)

        # check if number of clusters is less than number of selected molecules
        if not self.num_clusters > self.num_selected:
            raise ValueError("The number of clusters is great than number of selected molecules.")
        # check if we have valid number of clusters because selecting 1.5 molecules is not practical
        if int(self.num_selected / self.num_clusters) - self.num_selected / self.num_clusters != 0:
            raise ValueError("The number of molecules in each cluster should be an integer.")

        # todo: see if we need to add support of this set of algorithms
        # where we can combine soft clustering and Monte Carlo sampling

        # # check if selected number of clusters is less than the required number of molecules
        # if self.num_clusters > self.num_selected_mols:
        #     raise ValueError("The number of clusters cannot be greater than the number of "
        #                      "selected molecules.")
        #
        # # todo: check how many molecules/percents do we need reserve for sampling
        # # check if number of clusters is less than number of molecules when enhanced sampling is
        # # enabled
        # if enhanced_sampling_method is not None:
        #     if self.num_clusters >= self.num_selected_mols * self.enhanced_sampling_weight:
        #         raise ValueError(f"The defined number of clusters {self.num_clusters} is greater "
        #                          f"than {self.enhanced_sampling_weight} of number of selected"
        #                          f"molecules {self.num_selected_mols}.")

    def select(self):
        """
        Select the subset molecules based on the clustering method.
        :param num_selected: amount of molecule that need to be selected
        :param random_seed: int
        :return:
        """
        pass

    @property
    def subset_diversity(self):
        """
        Calculate diversity of the subset.
        :param subset: numpy array of indices which make up a subset
        :param div_metric: metric for calculating diversity of the subset ("Gini", "Entropy" etc.)
        :return: float #Scale should be discussed (0 to 1, or 0 to 100 or -1 to 1 or anything else)
        """
        # todo: need to implement diversity measurement here
        pass

    @property
    def all_diversity(self):
        """Original dataset diversity."""
        # todo: need to implement diversity measurement here
        return None
