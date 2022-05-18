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
"""MaxMin selector."""

import numpy as np
from DiverseSelector.utils import pick_initial_compounds
from DiverseSelector.base import SelectionBase


class MaxMin(SelectionBase):
    """Selecting compounds using MinMax algorithm."""
    def __init__(self):
        """Initializing class"""
        self.arr_dist = None
        self.n_mols = None

    @staticmethod
    def select_from_cluster(arr_dist, num_selected, indices=None):
        """
        MinMax algorithm for selecting points from cluster.

        Parameters
        ----------
        arr_dist: np.ndarray
            distance matrix for points that needs to be selected
        num_selected: int
            number of molecules that need to be selected
        indices: np.ndarray
            indices of molecules that form a cluster

        Returns
        -------
        selected: list
            list of ids of selected molecules

        """
        if indices is not None:
            arr_dist = arr_dist[indices][:, indices]
        selected = [pick_initial_compounds(arr_dist)]
        while len(selected) < num_selected:
            min_distances = np.min(arr_dist[selected], axis=0)
            new_id = np.argmax(min_distances)
            selected.append(new_id)
        return selected
