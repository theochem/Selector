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

from DiverseSelector.base import SelectionBase
from DiverseSelector.metric import ComputeDistanceMatrix
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

__all__ = [
    "DissimilaritySelection",
]


class DissimilaritySelection(SelectionBase):
    """Dissimilarity based diversity subset selection."""

    def __init__(self,
                 initialization="medoid",
                 metric="Tanimoto",
                 random_seed=42,
                 feature_type=None,
                 mol_file=None,
                 feature_file=None,
                 num_selected=None,
                 arr_dist=None,
                 method="maxmin",
                 r=1,
                 k=10,
                 cells=5,
                 max_dim=2,
                 grid_method="equisized_independent",
                 **kwargs,
                 ):
        """Initialization method for DissimilaritySelection class.

        Parameters
        ----------
        initialization
        metric
        random_seed
        feature_type
        mol_file
        feature_file
        num_selected
        arr_dist
        method
        r
        k
        cells
        max_dim
        grid_method
        kwargs
        """
        super().__init__(metric, random_seed, feature_type, mol_file, feature_file, num_selected)
        self.initialization = initialization
        self.arr_dist = arr_dist
        self.method = method
        self.r = r
        self.k = k
        self.cells = cells
        self.max_dim = max_dim
        self.grid_method = grid_method
        # super(DissimilaritySelection, self).__init__(**kwargs)
        self.__dict__.update(kwargs)

        # the initial compound index
        self.arr_dist, self.starting_idx = self.pick_initial_compounds()

    def pick_initial_compounds(self):
        """Pick the initial compounds."""
        # todo: current version only works for molecular descriptors
        # pair-wise distance matrix
        if self.arr_dist is None:
            dist_1 = ComputeDistanceMatrix(feature=self.features_norm,
                                           metric="euclidean")
            arr_dist_init = dist_1.compute_distance()

        # use the molecule with maximum distance to initial medoid as  the starting molecule
        if self.initialization.lower() == "medoid":
            # https://www.sciencedirect.com/science/article/abs/pii/S1093326399000145?via%3Dihub
            # J. Mol. Graphics Mod., 1998, Vol. 16,
            # DISSIM: A program for the analysis of chemical diversity
            medoid_idx = np.argmin(self.arr_dist.sum(axis=0))
            # selected molecule with maximum distance to medoid
            starting_idx = np.argmax(self.arr_dist[medoid_idx, :])
            arr_dist_init = self.arr_dist

        elif self.initialization.lower() == "random":
            rng = np.random.default_rng(self.random_seed)
            starting_idx = rng.choice(np.arange(self.features.shape[0]), 1)
            arr_dist_init = self.arr_dist

        return arr_dist_init, starting_idx

    def compute_diversity(self):
        """Compute the distance metrics."""
        # for iterative selection and final subset both
        pass

    def select(self, dissimilarity_function='brutestrength'):
        """Select method containing all dissimilarity algorithms.

        Parameters
        ----------
        dissimilarity_function

        Returns
        -------
        Chosen dissimilarity function.
        """
        def brutestrength(selected=None, n_selected=self.num_selected, method=self.method):
            """Brute Strength dissimilarity algorithm with maxmin and maxsum methods.

            Parameters
            ----------
            selected
            n_selected
            method

            Returns
            -------
            Selected molecules.
            """
            if selected is None:
                selected = [self.starting_idx]
                return brutestrength(selected, n_selected, method)

            # if we all selected all n_selected molecules then return list of selected mols
            if len(selected) == n_selected:
                return selected

            if method == 'maxmin':
                # calculate min distances from each mol to the selected mols
                min_distances = np.min(self.arr_dist[selected], axis=0)

                # find molecules distance minimal distance of which is the maximum among all
                new_id = np.argmax(min_distances)

                # add selected molecule to the selected list
                selected.append(new_id)

                # call method again with an updated list of selected molecules
                return brutestrength(selected, n_selected, method)
            elif method == 'maxsum':
                sum_distances = np.sum(self.arr_dist[selected], axis=0)
                while True:
                    new_id = np.argmax(sum_distances)
                    if new_id in selected:
                        sum_distances[new_id] = 0
                    else:
                        break
                selected.append(new_id)
                return brutestrength(selected, n_selected, method)
            else:
                raise ValueError(f"Method {method} not supported, choose maxmin or maxsum.")

        def gridpartitioning(selected=None, n_selected=self.num_selected, cells=self.cells,
                             max_dim=self.max_dim, array=self.features,
                             grid_method=self.grid_method):
            """Grid partitioning dissimilarity algorithm with various grid partitioning methods.

            Parameters
            ----------
            selected
            n_selected
            cells
            max_dim
            array
            grid_method

            Returns
            -------
            Selected molecules.
            """
            if selected is None:
                selected = []
                return gridpartitioning(selected, n_selected, cells, max_dim, array, grid_method)

            data_dim = len(array[0])
            if data_dim > max_dim:
                norm_data = StandardScaler().fit_transform(array)
                pca = PCA(n_components=max_dim)
                principalcomponents = pca.fit_transform(norm_data)
                return gridpartitioning(selected, n_selected, cells, max_dim,
                                        principalcomponents, grid_method)

            if grid_method == "equisized_independent":
                axis_info = []
                for i in range(data_dim):
                    axis_min, axis_max = min(array[:, i]), max(array[:, i])
                    cell_length = (axis_max - axis_min) / cells
                    axis_info.append([axis_min, axis_max, cell_length])
                bins = {}
                for index, point in enumerate(array):
                    point_bin = []
                    for dim, value in enumerate(point):
                        if value == axis_info[dim][0]:
                            index_bin = 0
                        elif value == axis_info[dim][1]:
                            index_bin = cells - 1
                        else:
                            index_bin = int((value - axis_info[dim][0]) // axis_info[dim][2])
                        point_bin.append(index_bin)
                    bins.setdefault(tuple(point_bin), [])
                    bins[tuple(point_bin)].append(index)

            elif grid_method == "equisized_dependent":
                bins = {}
                for i in range(data_dim):
                    if len(bins) == 0:
                        axis_min, axis_max = min(array[:, i]), max(array[:, i])
                        cell_length = (axis_max - axis_min) / cells
                        axis_info = [axis_min, axis_max, cell_length]

                        for index, point in enumerate(array):
                            point_bin = []
                            if point[i] == axis_info[0]:
                                index_bin = 0
                            elif point[i] == axis_info[1]:
                                index_bin = cells - 1
                            else:
                                index_bin = int((point[i] - axis_info[0]) // axis_info[2])
                            point_bin.append(index_bin)
                            bins.setdefault(tuple(point_bin), [])
                            bins[tuple(point_bin)].append(index)
                    else:
                        new_bins = {}
                        for bin_idx in bins:
                            axis_min = min(array[bins[bin_idx], i])
                            axis_max = max(array[bins[bin_idx], i])
                            cell_length = (axis_max - axis_min) / cells
                            axis_info = [axis_min, axis_max, cell_length]

                            for point_idx in bins[bin_idx]:
                                point_bin = [num for num in bin_idx]
                                if array[point_idx][i] == axis_info[0]:
                                    index_bin = 0
                                elif array[point_idx][i] == axis_info[1]:
                                    index_bin = cells - 1
                                else:
                                    index_bin = int((array[point_idx][i] - axis_info[0]) //
                                                    axis_info[2])
                                point_bin.append(index_bin)
                                new_bins.setdefault(tuple(point_bin), [])
                                new_bins[tuple(point_bin)].append(point_idx)
                        bins = new_bins

            elif grid_method == "equifrequent_independent":
                raise NotImplementedError(f"{grid_method} not implemented.")
            elif grid_method == "equifrequent_dependent":
                raise NotImplementedError(f"{grid_method} not implemented.")
            else:
                raise ValueError(f"{grid_method} not a valid method")

            old_len = 0
            rng = np.random.default_rng(seed=42)
            while len(selected) < n_selected:
                for bin_idx in bins:
                    if len(bins[bin_idx]) > 0:
                        random_int = rng.integers(low=0, high=len(bins[bin_idx]), size=1)[0]
                        mol_id = bins[bin_idx].pop(random_int)
                        selected.append(mol_id)

                if len(selected) == old_len:
                    break
                old_len = len(selected)
            return selected

        def sphereexclusion(selected=None, n_selected=12, s_max=1, order=None):
            """Directed sphere exclusion dissimilarity algorithm.

            Parameters
            ----------
            selected
            n_selected
            s_max
            order

            Returns
            -------
            Selected molecules.
            """
            if selected is None:
                selected = []
                return sphereexclusion(selected, n_selected, s_max, order)

            if order is None:
                ref = [self.starting_idx]
                candidates = np.delete(np.arange(0, len(self.features)), ref)
                distances = []
                for idx in candidates:
                    ref_point = self.features[ref[0]]
                    data_point = self.features[idx]
                    distance_sq = 0
                    for i, point in enumerate(ref_point):
                        distance_sq += (point - data_point[i]) ** 2
                    distances.append((distance_sq, idx))
                distances.sort()
                order = [idx for dist, idx in distances]
                return sphereexclusion(selected, n_selected, s_max, order)

            for idx in order:
                if len(selected) == 0:
                    selected.append(idx)
                    continue
                distances = []
                for selected_idx in selected:
                    data_point = self.features[idx]
                    selected_point = self.features[selected_idx]
                    distance_sq = 0
                    for i in range(len(data_point)):
                        distance_sq += (selected_point[i] - data_point[i]) ** 2
                    distances.append(np.sqrt(distance_sq))
                min_dist = min(distances)
                if min_dist > s_max:
                    selected.append(idx)
                if len(selected) == n_selected:
                    return selected

            return selected

        def optisim(selected=None, n_selected=self.num_selected, k=self.k,
                    r=self.r, recycling=None):
            """Optisim dissimilarity algorithm.

            Parameters
            ----------
            selected
            n_selected
            k
            r
            recycling

            Returns
            -------
            Selected molecules.
            """
            if selected is None:
                selected = [self.starting_idx]
                return optisim(selected, n_selected, k, r, recycling)

            if len(selected) >= n_selected:
                return selected

            if recycling is None:
                recycling = []

            candidates = np.delete(np.arange(0, len(self.features)), selected + recycling)
            subsample = {}
            while len(subsample) < k:
                if len(candidates) == 0:
                    if len(subsample) > 0:
                        selected.append(max(zip(subsample.values(), subsample.keys()))[1])
                        return optisim(selected, n_selected, k, r, recycling)
                    return selected
                rng = np.random.default_rng(seed=self.random_seed)
                random_int = rng.integers(low=0, high=len(candidates), size=1)[0]
                index_new = candidates[random_int]
                distances = []
                for selected_idx in selected:
                    data_point = self.features[index_new]
                    selected_point = self.features[selected_idx]
                    distance_sq = 0
                    for i, point in enumerate(data_point):
                        distance_sq += (selected_point[i] - point) ** 2
                    distances.append(np.sqrt(distance_sq))
                min_dist = min(distances)
                if min_dist > r:
                    subsample[index_new] = min_dist
                else:
                    recycling.append(index_new)
                candidates = np.delete(np.arange(0, len(self.features)),
                                       selected + recycling + list(subsample.keys()))
            selected.append(max(zip(subsample.values(), subsample.keys()))[1])

            return optisim(selected, n_selected, k, r, recycling)

        algorithms = {'brutestrength': brutestrength,
                      'gridpartitioning': gridpartitioning,
                      'sphereexclusion': sphereexclusion,
                      'optisim': optisim}
        return algorithms[dissimilarity_function]()
