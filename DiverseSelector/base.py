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

from .utils import feature_generator, feature_reader


class DissimilaritySelectionBase:
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
        self.features = self.get_features()

    def get_features(self):
        """Compute molecular features."""
        # todo: can be refactored to run faster

        # case: feature is not None, mol is None
        if self.mol_file is None and self.feature_file is not None:
            features = feature_reader(self.feature_file)
        # case: feature is None, mol is not None
        elif self.mol_file is not None and self.feature_file is None:
            features = feature_generator(self.mol_file)
        # case: feature is not None, mol is not None
        elif self.mol_file is not None and self.feature_file is not None:
            features = feature_reader(self.feature_file)
        # case: feature is None, mol is None
        else:
            raise ValueError("It is required to define the input molecule file or feature file.")

        return features

    def pick_initial_compounds(self):
        """Pick the initial compounds."""
        pass

    def compute_diversity(self):
        """Compute the distance metrics."""
        # for iterative selection and final subset both
        pass

    def select(self):
        """Select the subset molecules with optimal diversity."""
        pass

    @property
    def subset_diversity(self):
        """Selected subset diversity."""
        pass

    @property
    def all_diversity(self):
        """Original dataset diversity."""
        pass


class ClusteringSelectionBase(DissimilaritySelectionBase):
    def __init__(self,
                 clustering_method,
                 num_clusters=None,
                 enhanced_sampling_method=None,
                 enhanced_sampling_weight=None,
                 ):
        """Base class for clustering based subset selection."""
        self.clustering_method = clustering_method
        self.num_clusters = num_clusters
        self.enhanced_sampling_method = enhanced_sampling_method
        self.enhanced_sampling_weight = enhanced_sampling_weight
        super().__init__(self)

        # check if selected number of clusters is less than the required number of molecules
        if self.num_clusters > self.num_selected_mols:
            raise ValueError("The number of clusters cannot be greater than the number of "
                             "selected molecules.")

        # todo: check how many molecules/percents do we need reserve for sampling
        # check if number of clusters is less than number of molecules when enhanced sampling is
        # enabled
        if enhanced_sampling_method is not None:
            if self.num_clusters >= self.num_selected_mols * self.enhanced_sampling_weight:
                raise ValueError(f"The defined number of clusters {self.num_clusters} is greater "
                                 f"than {self.enhanced_sampling_weight} of number of selected"
                                 f"molecules {self.num_selected_mols}.")
