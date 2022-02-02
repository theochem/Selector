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
from .utils import feature_generator, feature_reader, get_features


class DissimilaritySelectionBase(ABC):
    def __init__(self,
                 num_selected,
                 initialization="medoid",
                 metric="Tanimoto",
                 feature_file=None,
                 mol_file=None,
                 output=None,
                 random_seed=None,
                 ):
        """Base class for dissimilarity based subset selection."""
        self.num_selected = num_selected
        self.initialization = initialization
        self.metric = metric
        self.output = output
        self.feature_file = feature_file
        self.mol_file = mol_file
        if random_seed is None:
            self.random_seed = 42
        else:
            self.random_seed = random_seed

        # compute/load molecular features
        self.features = get_features(mol_file,
                                     feature_file)

    @abstractmethod
    def pick_initial_compounds(self):
        """Pick the initial compounds."""
        pass

    @abstractmethod
    def compute_diversity(self):
        """Compute the distance metrics."""
        # for iterative selection and final subset both
        pass

    @abstractmethod
    def select(self):
        """Select the subset molecules with optimal diversity."""
        pass

    @property
    def subset_diversity(self):
        """Selected subset diversity."""
        # todo: need to implement diversity measurement here
        return None

    @property
    def all_diversity(self):
        """Original dataset diversity."""
        # todo: need to implement diversity measurement here
        return None


class ClusteringSelectionBase(ABC):
    def __init__(self,
                 clustering_method,
                 metric="Tanimoto",
                 num_selected=None,
                 num_clusters=None,
                 mol_file=None,
                 feature_file=None,
                 output=None,
                 random_seed=None,
                 **kwargs
                 ):
        """Base class for clustering based subset selection."""

        # the number of molecules equals the number of clusters
        self.clustering_method = clustering_method
        self.metric = metric
        self.num_selected = num_selected
        self.num_clusters = num_clusters
        self.mol_file = mol_file
        self.feature_file = feature_file
        self.output = output

        if random_seed is None:
            self.random_seed = 42
        else:
            self.random_seed = random_seed

        self.__dict__.update(kwargs)

        # check if number of clusters is less than number of selected molecules
        if not num_clusters > num_selected:
            raise ValueError("The number of clusters is great than number of selected molecules.")
        # check if we have valid number of clusters because selecting 1.5 molecules is not practical
        if int(num_selected / num_clusters) - num_selected / num_clusters != 0:
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

    # def enhanced_sampling(self):
    #     """Enhanced sampling with random sampling or Monte Carlo sampling."""
    #     pass

    @abstractmethod
    def select(self):
        """Add support of clustering and enhanced sampling."""
        pass

    @property
    def subset_diversity(self):
        """Selected subset diversity."""
        # todo: need to implement diversity measurement here
        return None

    @property
    def all_diversity(self):
        """Original dataset diversity."""
        # todo: need to implement diversity measurement here
        return None

# class ClusteringSelectionBase(DissimilaritySelectionBase, ABC):
#     def __init__(self,
#                  clustering_method,
#                  num_clusters=None,
#                  enhanced_sampling_method=None,
#                  enhanced_sampling_weight=None,
#                  ):
#         """Base class for clustering based subset selection."""
#         self.clustering_method = clustering_method
#         self.num_clusters = num_clusters
#         self.enhanced_sampling_method = enhanced_sampling_method
#         self.enhanced_sampling_weight = enhanced_sampling_weight
#         super().__init__(self)
#
#         # check if selected number of clusters is less than the required number of molecules
#         if self.num_clusters > self.num_selected_mols:
#             raise ValueError("The number of clusters cannot be greater than the number of "
#                              "selected molecules.")
#
#         # todo: check how many molecules/percents do we need reserve for sampling
#         # check if number of clusters is less than number of molecules when enhanced sampling is
#         # enabled
#         if enhanced_sampling_method is not None:
#             if self.num_clusters >= self.num_selected_mols * self.enhanced_sampling_weight:
#                 raise ValueError(f"The defined number of clusters {self.num_clusters} is greater "
#                                  f"than {self.enhanced_sampling_weight} of number of selected"
#                                  f"molecules {self.num_selected_mols}.")
#
#     def enhanced_sampling(self):
#         """Enhanced sampling with random sampling or Monte Carlo sampling."""
#         pass
#
#     def select(self):
#         """Add support of clustering and enhanced sampling."""
#         pass
