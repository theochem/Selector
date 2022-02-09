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

"""Dissimilarity based diversity subset selection."""

import numpy as np
from .base import SelectionBase
from .metric import pairwise_dist
from .utils import get_features


class DissimilaritySelection(SelectionBase):

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
        super().__init__(metric, random_seed, feature_type, mol_file, feature_file, num_selected, **kwargs)
        self.initialization = initialization


        # super(DissimilaritySelection, self).__init__(**kwargs)  NOTE:# I'm not sure what it should do
        self.__dict__.update(kwargs)

        # the initial compound index
        self.arr_dist, self.starting_idx = self.pick_initial_compounds()

    def pick_initial_compounds(self):
        """Pick the initial compounds."""
        # todo: current version only works for molecular descriptors
        # pair-wise distance matrix
        arr_dist = pairwise_dist(feature=self.features_norm,
                                 metric="euclidean")

        # use the molecule with maximum distance to initial medoid as  the starting molecule
        if self.initialization.lower() == "medoid":
            # https://www.sciencedirect.com/science/article/abs/pii/S1093326399000145?via%3Dihub
            # J. Mol. Graphics Mod., 1998, Vol. 16,
            # DISSIM: A program for the analysis of chemical diversity
            medoid_idx = np.argmin(arr_dist.sum(axis=0))
            # selected molecule with maximum distance to medoid 
            starting_idx = np.argmax(arr_dist[medoid_idx, :])

        elif self.initialization.lower() == "random":
            rng = np.random.default_rng(self.random_seed)
            starting_idx = rng.choice(np.arange(self.features_norm.shape[0]), 1)

        return arr_dist, starting_idx

    def compute_diversity(self):
        """Compute the distance metrics."""
        # for iterative selection and final subset both
        pass

    def select(self):
        """Select the subset molecules with optimal diversity."""
        # a 1d vector to store the index of selected molecules
        selected_indices = np.full((self.num_selected,), False)
        selected_indices[self.starting_idx] = True

        arr_dist_new = self.arr_dist.copy()

        # todo: mimmax algorithm implementation
        # for idx_counter in np.arange(1, self.num_selected):
        pass
