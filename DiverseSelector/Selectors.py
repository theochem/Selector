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

"""Selectors classes for different choices of subset selection."""


import numpy as np
from DiverseSelector.utils import pick_initial_compounds
from DiverseSelector.base import SelectionBase


class MaxMin(SelectionBase):
    """Selecting compounds using MinMax algorithm."""
    def __init__(self, func_distance=None):
        """Initializing class"""
        self.arr_dist = None
        self.n_mols = None
        self.func_distance = func_distance

    def select_from_cluster(self, arr, num_selected, indices=None):
        """
        MinMax algorithm for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            distance matrix for points that needs to be selected if func_distance is None.
            Otherwise, treated as coordinates array.
        num_selected: int
            number of molecules that need to be selected
        indices: np.ndarray
            indices of molecules that form a cluster

        Returns
        -------
        selected: list
            list of ids of selected molecules

        """
        self.n_mols = arr.shape[0]
        if self.func_distance is not None:
            self.arr_dist = self.func_distance(arr)
        else:
            self.arr_dist = arr

        if indices is not None:
            arr_dist = self.arr_dist[indices][:, indices]
        else:
            arr_dist = self.arr_dist
        selected = [pick_initial_compounds(arr_dist)]
        while len(selected) < num_selected:
            min_distances = np.min(arr_dist[selected], axis=0)
            new_id = np.argmax(min_distances)
            selected.append(new_id)
        return selected


class OptiSim(SelectionBase):
    def __init__(self, r=None, k=10):
        self.r = r
        self.k = k
        self.random_seed = 42
        self.starting_idx = 0

    def optisim(self, arr):
        selected = [self.starting_idx]
        recycling = []

        candidates = np.delete(np.arange(0, len(arr)), selected + recycling)
        subsample = {}
        while len(candidates) > 0:
            while len(subsample) < self.k:
                if len(candidates) == 0:
                    if len(subsample) > 0:
                        break
                    return selected
                rng = np.random.default_rng(seed=self.random_seed)
                random_int = rng.integers(low=0, high=len(candidates), size=1)[0]
                index_new = candidates[random_int]
                distances = []
                for selected_idx in selected:
                    data_point = arr[index_new]
                    selected_point = arr[selected_idx]
                    distance_sq = 0
                    for i, point in enumerate(data_point):
                        distance_sq += (selected_point[i] - point) ** 2
                    distances.append(np.sqrt(distance_sq))
                min_dist = min(distances)
                if min_dist > self.r:
                    subsample[index_new] = min_dist
                else:
                    recycling.append(index_new)
                candidates = np.delete(np.arange(0, len(arr)),
                                       selected + recycling + list(subsample.keys()))
            selected.append(max(zip(subsample.values(), subsample.keys()))[1])
            candidates = np.delete(np.arange(0, len(arr)), selected + recycling)
            subsample = {}

        return selected

    def select_from_cluster(self, arr, num_select, indices=None):
        if indices is not None:
            arr = arr[indices]
        if self.r is None:
            # Use numpy.optimize.bisect instead
            arr_range = (max(arr[:, 0]) - min(arr[:, 0]),
                         max(arr[:, 1]) - min(arr[:, 1]))
            rg = max(arr_range) / num_select * 3
            self.r = rg
            result = self.optisim(arr)
            if len(result) == num_select:
                return result

            low = rg if len(result) > num_select else 0
            high = rg if low == 0 else None
            bounds = [low, high]
            count = 0
            while len(result) != num_select and count < 20:
                if bounds[1] is None:
                    rg = bounds[0] * 2
                else:
                    rg = (bounds[0] + bounds[1]) / 2
                self.r = rg
                result = self.optisim(arr)
                if len(result) > num_select:
                    bounds[0] = rg
                else:
                    bounds[1] = rg
                count += 1
            self.r = None
            return result
        else:
            return self.optisim(arr)


class DirectedSphereExclusion(SelectionBase):
    def __init__(self, r=None):
        self.r = r
        self.random_seed = 42
        self.starting_idx = 0

    def sphere_exclusion(self, arr):
        selected = []
        ref = [self.starting_idx]
        candidates = np.delete(np.arange(0, len(arr)), ref)
        distances = []
        for idx in candidates:
            ref_point = arr[ref[0]]
            data_point = arr[idx]
            distance_sq = 0
            for i, point in enumerate(ref_point):
                distance_sq += (point - data_point[i]) ** 2
            distances.append((distance_sq, idx))
        distances.sort()
        order = [idx for dist, idx in distances]

        for idx in order:
            if len(selected) == 0:
                selected.append(idx)
                continue
            distances = []
            for selected_idx in selected:
                data_point = arr[idx]
                selected_point = arr[selected_idx]
                distance_sq = 0
                for i, point in enumerate(data_point):
                    distance_sq += (selected_point[i] - point) ** 2
                distances.append(np.sqrt(distance_sq))
            min_dist = min(distances)
            if min_dist > self.r:
                selected.append(idx)

        return selected

    def select_from_cluster(self, arr, num_selected, indices=None):
        if indices is not None:
            arr = arr[indices]
        if self.r is None:
            # Use numpy.optimize.bisect instead
            arr_range = (max(arr[:, 0]) - min(arr[:, 0]),
                         max(arr[:, 1]) - min(arr[:, 1]))
            rg = max(arr_range) / num_selected * 3
            self.r = rg
            result = self.sphere_exclusion(arr)
            if len(result) == num_selected:
                return result

            low = rg if len(result) > num_selected else 0
            high = rg if low == 0 else None
            bounds = [low, high]
            count = 0
            while len(result) != num_selected and count < 20:
                if bounds[1] is None:
                    rg = bounds[0] * 2
                else:
                    rg = (bounds[0] + bounds[1]) / 2
                self.r = rg
                result = self.sphere_exclusion(arr)
                if len(result) > num_selected:
                    bounds[0] = rg
                else:
                    bounds[1] = rg
                count += 1
            self.r = None
            return result
        else:
            return self.sphere_exclusion(arr)
