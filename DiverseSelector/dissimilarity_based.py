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
from pathlib import PurePath
from typing import Union

from DiverseSelector.base import SelectionBase
from DiverseSelector.utils import PandasDataFrame
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

__all__ = [
    "DissimilaritySelection",
]


class DissimilaritySelection(SelectionBase):
    """Dissimilarity based diversity subset selection."""

    def __init__(self,
                 features: Union[np.ndarray, PandasDataFrame, str, PurePath] = None,
                 arr_dist: np.ndarray = None,
                 normalize_features: bool = False,
                 sep: str = ",",
                 engine: str = "python",
                 initialization="medoid",
                 random_seed=42,
                 num_selected: int = None,
                 dissim_func="brute_strength",
                 brute_strength_type="maxmin",
                 r=1,
                 k=10,
                 cells=5,
                 max_dim=2,
                 grid_method="equisized_independent",
                 **kwargs,
                 ):
        """Initialization brute_strength_type for DissimilaritySelection class.

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
        brute_strength_type
        r
        k
        cells
        max_dim
        grid_method
        kwargs
        """

        super().__init__(features,
                         arr_dist,
                         num_selected,
                         normalize_features,
                         sep,
                         engine,
                         random_seed,
                         **kwargs,
                         )
        self.initialization = initialization
        self.r = r
        self.k = k
        self.cells = cells
        self.max_dim = max_dim
        self.grid_method = grid_method
        self.dissim_func = dissim_func
        self.brute_strength_type = brute_strength_type
        # super(DissimilaritySelection, self).__init__(**kwargs)
        # self.__dict__.update(kwargs)

        # data type checking
        if self.dissim_func in ["grid_partitioning",
                                "sphere_exclusion",
                                "optisim",
                                ]:
            # feature is required for grid partitioning methods
            if self.features is None:
                raise ValueError(f"Features must be provided for {self.dissim_func} method.")
            # convert pandas dataframe to numpy array
            if isinstance(self.features, pd.DataFrame):
                self.features = self.features.to_numpy()

        # the initial compound index
        self.starting_idx = self.pick_initial_compounds()

    def pick_initial_compounds(self):
        """Pick the initial compounds."""

        # use the molecule with maximum distance to initial medoid as  the starting molecule
        if self.initialization.lower() == "medoid":
            # https://www.sciencedirect.com/science/article/abs/pii/S1093326399000145?via%3Dihub
            # J. Mol. Graphics Mod., 1998, Vol. 16,
            # DISSIM: A program for the analysis of chemical diversity
            medoid_idx = np.argmin(self.arr_dist.sum(axis=0))
            # selected molecule with maximum distance to medoid
            starting_idx = np.argmax(self.arr_dist[medoid_idx, :])

        elif self.initialization.lower() == "random":
            rng = np.random.default_rng(seed=self.random_seed)
            starting_idx = rng.choice(np.arange(self.features.shape[0]), 1)
        else:
            raise ValueError(f"Initialization method {self.initialization} is not supported.")

        return starting_idx

    def compute_diversity(self):
        """Compute the distance metrics."""
        # for iterative selection and final subset both
        pass

    def select(self):
        """Select brute_strength_type containing all dissimilarity algorithms.

        Parameters
        ----------
        dissim_func

        Returns
        -------
        Chosen dissimilarity function.
        """
        def brute_strength(selected=None,
                           n_selected=self.num_selected,
                           brute_strength_type=self.brute_strength_type,
                           ):
            """Brute Strength dissimilarity algorithm with maxmin and maxsum methods.

            Parameters
            ----------
            selected
            n_selected
            brute_strength_type

            Returns
            -------
            Selected molecules.
            """
            if selected is None:
                selected = [self.starting_idx]
                return brute_strength(selected, n_selected, brute_strength_type)

            # if we all selected all n_selected molecules then return list of selected mols
            if len(selected) == n_selected:
                return selected

            if brute_strength_type == "maxmin":
                # calculate min distances from each mol to the selected mols
                min_distances = np.min(self.arr_dist[selected], axis=0)

                # find molecules distance minimal distance of which is the maximum among all
                new_id = np.argmax(min_distances)

                # add selected molecule to the selected list
                selected.append(new_id)

                # call brute_strength_type again with an updated list of selected molecules
                return brute_strength(selected, n_selected, brute_strength_type)
            elif brute_strength_type == "maxsum":
                sum_distances = np.sum(self.arr_dist[selected], axis=0)
                while True:
                    new_id = np.argmax(sum_distances)
                    if new_id in selected:
                        sum_distances[new_id] = 0
                    else:
                        break
                selected.append(new_id)
                return brute_strength(selected, n_selected, brute_strength_type)
            else:
                raise ValueError(f"""Method {brute_strength_type} is not supported, choose """
                                 f"""maxmin" or "maxsum".""")

        def grid_partitioning(selected=None,
                              n_selected=self.num_selected,
                              cells=self.cells,
                              max_dim=self.max_dim,
                              arr_features=self.features,
                              grid_method=self.grid_method,
                              ):
            """Grid partitioning dissimilarity algorithm with various grid partitioning methods.

            Parameters
            ----------
            selected
            n_selected
            cells
            max_dim
            arr_features
            grid_method

            Returns
            -------
            Selected molecules.
            """

            if selected is None:
                selected = []
                return grid_partitioning(selected, n_selected, cells, max_dim, arr_features, grid_method)

            data_dim = len(arr_features[0])
            if data_dim > max_dim:
                norm_data = StandardScaler().fit_transform(arr_features)
                pca = PCA(n_components=max_dim)
                principal_components = pca.fit_transform(norm_data)
                return grid_partitioning(selected, n_selected, cells, max_dim,
                                         principal_components, grid_method)

            if grid_method == "equisized_independent":
                axis_info = []
                for i in range(data_dim):
                    axis_min, axis_max = min(arr_features[:, i]), max(arr_features[:, i])
                    cell_length = (axis_max - axis_min) / cells
                    axis_info.append([axis_min, axis_max, cell_length])
                bins = {}
                for index, point in enumerate(arr_features):
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
                        axis_min, axis_max = min(arr_features[:, i]), max(arr_features[:, i])
                        cell_length = (axis_max - axis_min) / cells
                        axis_info = [axis_min, axis_max, cell_length]

                        for index, point in enumerate(arr_features):
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
                        for bin_idx, bin_list in bins.items():
                            axis_min = min(arr_features[bin_list, i])
                            axis_max = max(arr_features[bin_list, i])
                            cell_length = (axis_max - axis_min) / cells
                            axis_info = [axis_min, axis_max, cell_length]

                            for point_idx in bin_list:
                                point_bin = [num for num in bin_idx]
                                if arr_features[point_idx][i] == axis_info[0]:
                                    index_bin = 0
                                elif arr_features[point_idx][i] == axis_info[1]:
                                    index_bin = cells - 1
                                else:
                                    index_bin = int((arr_features[point_idx][i] - axis_info[0]) //
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
                raise ValueError(f"{grid_method} not a valid brute_strength_type")

            old_len = 0
            rng = np.random.default_rng(seed=self.random_seed)
            while len(selected) < n_selected:
                for bin_idx, bin_list in bins.items():
                    if len(bin_list) > 0:
                        random_int = rng.integers(low=0, high=len(bin_list), size=1)[0]
                        mol_id = bin_list.pop(random_int)
                        selected.append(mol_id)

                if len(selected) == old_len:
                    break
                old_len = len(selected)
            return selected

        def sphere_exclusion(selected=None,
                             n_selected=12,
                             s_max=1,
                             order=None,
                             ):
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
                return sphere_exclusion(selected, n_selected, s_max, order)

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
                return sphere_exclusion(selected, n_selected, s_max, order)

            for idx in order:
                if len(selected) == 0:
                    selected.append(idx)
                    continue
                distances = []
                for selected_idx in selected:
                    data_point = self.features[idx]
                    selected_point = self.features[selected_idx]
                    distance_sq = 0
                    for i, point in enumerate(data_point):
                        distance_sq += (selected_point[i] - point) ** 2
                    distances.append(np.sqrt(distance_sq))
                min_dist = min(distances)
                if min_dist > s_max:
                    selected.append(idx)
                if len(selected) == n_selected:
                    return selected

            return selected

        def optisim(selected=None,
                    n_selected=self.num_selected,
                    k=self.k,
                    r=self.r,
                    recycling=None,
                    ):
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

        select_algorithms = {"brute_strength": brute_strength,
                             "grid_partitioning": grid_partitioning,
                             "sphere_exclusion": sphere_exclusion,
                             "optisim": optisim}
        return select_algorithms[self.dissim_func]()
