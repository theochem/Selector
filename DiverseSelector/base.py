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

import numpy as np

from .utils import feature_generator, feature_reader, get_features

class Selector(ABC):
    @abstractmethod
    def pick_initial_compounds(self):
        """Pick the initial compounds.
        """
        pass

    @abstractmethod
    def compute_diversity(self, metric: str, func=None):
        """
        Compute the diversity between every component in dataset
        :param metric: one of the standard metrics ("Euclidean", "Cosine", "Jaccard" etc. )
        :param func: callable function that take 2 arguments and calculate distance between them. Specified by user
        :return: np.ndarray aka (dis-)similarity matrix
        """
        pass

    @abstractmethod
    def select(self, num_selected: int, random_seed: int = None):
        """
        Select the subset molecules with optimal diversity.
        :param num_selected: amount of molecule that need to be selected
        :param random_seed: random seed for random selection compounds
        :return: np.array of indices of selected molecules from the initial dataset ## Maybe not indices....
        """
        pass

    @abstractmethod
    def subset_diversity(self, subset: np.ndarray, div_metric: str):
        """
        Calculate diversity of the subset.
        :param subset: numpy array of indices which make up a subset
        :param div_metric: metric for calculating diversity of the subset ("Gini", "Entropy" etc.)
        :return: float #Scale should be discussed (0 to 1, or 0 to 100 or -1 to 1 or anything else)
        """
        # todo: need to implement diversity measurement here
        pass

    @abstractmethod
    @property
    def all_diversity(self, div_metric: str): # Not an abstract method!!! Needs to be implemented explicitly here
        """
        Calculate diversity of the original dataset.
        :param div_metric: metric for calculating diversity of the subset ("Gini", "Entropy" etc.)
        :return: float #Scale should be discussed (0 to 1, or 0 to 100 or -1 to 1 or anything else)
        """
        # todo: need to implement diversity measurement here
        pass

    def read_excel(self, fname: str):  # Not an abstract method!!! Needs to be implemented explicitly here
        """
        Read dataset from an excel file
        :param fname: file name for saving molecule
        :return: None
        """
        pass

    def to_excel(self, fname: str):  # Not an abstract method!!! Needs to be implemented explicitly here
        """
        Save selected molecules as an excel file
        :param fname: file name for saving molecule
        :return: None
        """
        pass

    def to_txt(self, fname: str): # Not an abstract method!!! Needs to be implemented explicitly here
        """
        Save selected molecules as an excel file
        :param fname: file name for saving molecule
        :return: None
        """
        pass


class DissimilaritySelectionBase(Selector):
    def __init__(self,
                 initialization="medoid",
                 metric="Tanimoto",
                 feature_file=None):
        """Base class for dissimilarity based subset selection."""
        self.initialization = initialization
        self.metric = metric
        self.feature_file = feature_file
        # compute/load molecular features
        self.features = get_features(mol_file,
                                     feature_file)

    def pick_initial_compounds(self):
        """Pick the initial compounds."""
        pass

    def compute_diversity(self):
        """Compute the distance metrics."""
        # for iterative selection and final subset both
        pass

    def select(self, num_selected: int, random_seed = None):
        """
        Select the subset molecules with optimal diversity.
        :param num_selected: amount of molecule that need to be selected
        :param random_seed: int
        :return:
        """
        self.random_seed = 42 if random_seed is None else random_seed
        pass



class ClusteringSelectionBase(ABC):
    def __init__(self,
                 metric="Tanimoto",
                 num_clusters=None,
                 feature_file=None,
                 output=None,
                 random_seed=None,
                 **kwargs
                 ):
        """Base class for clustering based subset selection."""

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

    def cluster(self, clustering_method:str):
        """
        Perform the clastering of the dataset
        :param clustering_method: selected method for clustering (k-means, DBSCAN etc)
        :return:
        """
        pass

    def select(self, num_selected: int, random_seed=None):
        """
        Select the subset molecules with optimal diversity.
        :param num_selected: amount of molecule that need to be selected
        :param random_seed: int
        :return:
        """
        self.random_seed = 42 if random_seed is None else random_seed
        # check if number of clusters is less than number of selected molecules
        if not self.num_clusters > num_selected:
            raise ValueError("The number of clusters is great than number of selected molecules.")
        # check if we have valid number of clusters because selecting 1.5 molecules is not practical
        if int(num_selected / num_clusters) - num_selected / num_clusters != 0:
            raise ValueError("The number of molecules in each cluster should be an integer.")
        pass

    def subset_diversity(self, subset: np.ndarray, div_metric: str):
        """
        Calculate diversity of the subset.
        :param subset: numpy array of indices which make up a subset
        :param div_metric: metric for calculating diversity of the subset ("Gini", "Entropy" etc.)
        :return: float #Scale should be discussed (0 to 1, or 0 to 100 or -1 to 1 or anything else)
        """
        # todo: need to implement diversity measurement here
        pass

    def all_diversity(self):
        """Original dataset diversity."""
        # todo: need to implement diversity measurement here
        return None
