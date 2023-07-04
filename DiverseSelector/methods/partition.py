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

"""Module for Partition-Based Selection Methods."""

import collections
import math

import bitarray
import scipy.spatial

from DiverseSelector.methods.base import SelectionBase
from DiverseSelector.diversity import compute_diversity
from DiverseSelector.methods.utils import predict_radius
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


__all__ = [
    "DirectedSphereExclusion",
    "GridPartitioning",
    "Medoid",
]


class DirectedSphereExclusion(SelectionBase):
    """Selecting points using Directed Sphere Exclusion algorithm.

    Starting point is chosen as the reference point
    and not included in the selected molecules. The
    distance of each point is calculated to the reference point and the points are then sorted based
    on the ascending order of distances. The points are then evaluated in their sorted order, and
    are selected if their distance to all the other selected points is at least r away. Euclidean
    distance is used by default and the r value is automatically generated if not passed to satisfy
    the number of molecules requested.

    Notes
    -----
    Gobbi, A., and Lee, M.-L. (2002). DISE: directed sphere exclusion.
    Journal of Chemical Information and Computer Sciences,
    43(1), 317–323. https://doi.org/10.1021/ci025554v
    """

    def __init__(self, r=None, tolerance=0.05, eps=1e-8, p=2, start_id=0, random_seed=42):
        """
        Initializing class.

        Parameters
        ----------
        r: float
            Initial guess of radius for directed sphere exclusion algorithm, no points within r
            distance to an already selected point can be selected.
        tolerance: float
            Percentage error of number of points actually selected from number of points
            requested.
        eps: float
            Approximate nearest neighbor search for eliminating close points. Branches of the tree
            are not explored if their nearest points are further than r / (1 + eps), and branches
            are added in bulk if their furthest points are nearer than r * (1 + eps).
        p: float
            Which Minkowski p-norm to use. Should be in the range [1, inf]. A finite large p may
            cause a ValueError if overflow can occur.
        start_id: int
            Index for the first point to be selected.
        random_seed: int
            Seed for random selection of points be evaluated.
        """
        self.r = r
        self.tolerance = tolerance
        self.eps = eps
        self.p = p
        self.starting_idx = start_id
        self.random_seed = random_seed

    def algorithm(self, x, uplimit):
        """
        Directed sphere exclusion algorithm.

        Given a reference point, sorts all points by distance to the reference point.
        Then using a KDTree, the closest points are selected and a sphere
        is built around the point. All points within the sphere are excluded
        from the search. This process iterates until the number of selected
        points is greater than `uplimit`, or the algorithm runs out of points
        to select from.

        Parameters
        ----------
        x: np.ndarray
            Feature matrix.
        uplimit: int
            Maximum number of points to select.

        Returns
        -------
        selected: list
            List of ids of selected points.
        """
        selected = []
        count = 0
        # calculate distance from reference point to all data points
        ref_point = x[self.starting_idx]
        distances = scipy.spatial.minkowski_distance(ref_point, x, p=self.p)
        # order points by distance from reference
        order = np.argsort(distances)
        # Construct KDTree to make it easier to search neighbors
        kdtree = spatial.KDTree(x)
        # bv tracks viable candidates
        bv = bitarray.bitarray(len(x))
        bv[:] = 0
        bv[self.starting_idx] = 1
        # select points based on closest to reference point
        for idx in order:
            # If it isn't already part of any hyperspheres
            if not bv[idx]:
                # Then select it to be part of it
                selected.append(idx)
                count += 1
                # finished selecting # of points required, return
                if count > uplimit:
                    return selected
                # find all points now within radius of newly selected point
                elim = kdtree.query_ball_point(x[idx], self.r, eps=self.eps, p=self.p, workers=-1)
                # turn 'on' bits in bv to make for loop skip indices of eliminated points
                #   eliminate points from selection
                for index in elim:
                    bv[index] = 1

        return selected

    def select_from_cluster(self, x, num_selected, cluster_ids=None):
        """
        Algorithm that uses sphere_exclusion for selecting points from cluster.

        Parameters
        ----------
        x: np.ndarray
            Feature points.
        num_selected: int
            Number of points that need to be selected.
        cluster_ids: np.ndarray
            Indices of points that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if x.shape[0] < num_selected:
            raise RuntimeError(
                f"The number of selected points {num_selected} is greater than the number of points"
                f"provided {x.shape[0]}."
            )
        return predict_radius(self, x, num_selected, cluster_ids)


class GridPartitioning(SelectionBase):
    """Selecting points using the Grid Partitioning algorithm.

    Points are partitioned into grids using an algorithm (equisized independent or equisized
    dependent). A point is selected from each of the grids while the number of selected points is
    less than the number requested and while the grid has available points remaining, looping until
    the number of requested points is satisfied. If at the end, the number of points needed is less
    than the number of grids available to select from, the points are chosen from the grids with the
    greatest diversity.

    Adapted from https://doi.org/10.1016/S1093-3263(99)00016-9.
    """

    def __init__(self, cells, grid_method="equisized_independent", max_dim=None, random_seed=42):
        """
        Initializing class.

        Parameters
        ----------
        cells: int
            Number of cells to partition each axis into, the number of resulting grids is cells to
            the power of the dimensionality of the coordinate array.
        grid_method: str
            Grid method used to partition the points into grids. "equisized_independent" and
            "equisized_dependent" are supported options.
        max_dim: int
            Maximum dimensionality of coordinate array, if the dimensionality is greater than the
            max_dim provided then dimensionality reduction is done using PCA.
        random_seed: int
            Seed for random selection of points to be selected from each grid.
        """
        self.random_seed = random_seed
        self.cells = cells
        self.max_dim = max_dim
        self.grid_method = grid_method

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Grid partitioning algorithm for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points
        num_selected: int
            Number of molecules that need to be selected.
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if cluster_ids is not None:
            arr = arr[cluster_ids]

        selected = []
        data_dim = len(arr[0])
        if self.max_dim is not None and data_dim > self.max_dim:
            norm_data = StandardScaler().fit_transform(arr)
            pca = PCA(n_components=self.max_dim)
            arr = pca.fit_transform(norm_data)
            data_dim = self.max_dim

        if self.grid_method == "equisized_independent":
            axis_info = []
            for i in range(data_dim):
                axis_min, axis_max = min(arr[:, i]), max(arr[:, i])
                cell_length = (axis_max - axis_min) / self.cells
                axis_info.append([axis_min, axis_max, cell_length])
            bins = {}
            for index, point in enumerate(arr):
                point_bin = []
                for dim, value in enumerate(point):
                    if value == axis_info[dim][0]:
                        index_bin = 0
                    elif value == axis_info[dim][1]:
                        index_bin = self.cells - 1
                    else:
                        index_bin = int((value - axis_info[dim][0]) // axis_info[dim][2])
                    point_bin.append(index_bin)
                bins.setdefault(tuple(point_bin), [])
                bins[tuple(point_bin)].append(index)

        elif self.grid_method == "equisized_dependent":
            bins = {}
            for i in range(data_dim):
                if len(bins) == 0:
                    axis_min, axis_max = min(arr[:, i]), max(arr[:, i])
                    cell_length = (axis_max - axis_min) / self.cells
                    axis_info = [axis_min, axis_max, cell_length]

                    for index, point in enumerate(arr):
                        point_bin = []
                        if point[i] == axis_info[0]:
                            index_bin = 0
                        elif point[i] == axis_info[1]:
                            index_bin = self.cells - 1
                        else:
                            index_bin = int((point[i] - axis_info[0]) // axis_info[2])
                        point_bin.append(index_bin)
                        bins.setdefault(tuple(point_bin), [])
                        bins[tuple(point_bin)].append(index)
                else:
                    new_bins = {}
                    for bin_idx, bin_list in bins.items():
                        axis_min = min(arr[bin_list, i])
                        axis_max = max(arr[bin_list, i])
                        cell_length = (axis_max - axis_min) / self.cells
                        axis_info = [axis_min, axis_max, cell_length]

                        for point_idx in bin_list:
                            point_bin = [num for num in bin_idx]
                            if arr[point_idx][i] == axis_info[0]:
                                index_bin = 0
                            elif arr[point_idx][i] == axis_info[1]:
                                index_bin = self.cells - 1
                            else:
                                index_bin = int((arr[point_idx][i] - axis_info[0]) // axis_info[2])
                            point_bin.append(index_bin)
                            new_bins.setdefault(tuple(point_bin), [])
                            new_bins[tuple(point_bin)].append(point_idx)
                    bins = new_bins

        elif self.grid_method == "equifrequent_independent":
            raise NotImplementedError(f"{self.grid_method} not implemented.")
        elif self.grid_method == "equifrequent_dependent":
            raise NotImplementedError(f"{self.grid_method} not implemented.")
        else:
            raise ValueError(f"{self.grid_method} not a valid grid_method")

        old_len = 0
        to_delete = []
        rng = np.random.default_rng(seed=self.random_seed)
        while len(selected) < num_selected:
            num_needed = num_selected - len(selected)
            bin_count = len(bins)
            if bin_count <= num_needed:
                for bin_idx, bin_list in bins.items():
                    random_int = rng.integers(low=0, high=len(bin_list), size=1)[0]
                    mol_id = bin_list.pop(random_int)
                    selected.append(mol_id)
                    if len(bin_list) == 0:
                        to_delete.append(bin_idx)
                for idx in to_delete:
                    del bins[idx]
                to_delete = []

            else:
                diversity = []
                for bin_idx, bin_list in bins.items():
                    diversity.append((compute_diversity(arr[bin_list]), bin_idx))
                diversity.sort(reverse=True)
                for _, bin_idx in diversity[:num_needed]:
                    random_int = rng.integers(low=0, high=len(bins[bin_idx]), size=1)[0]
                    mol_id = bins[bin_idx].pop(random_int)
                    selected.append(mol_id)
            if len(selected) == old_len:
                break
            old_len = len(selected)
        return selected


class Medoid(SelectionBase):
    """Selecting points using an algorithm adapted from KDTree.

    Points are initially used to construct a KDTree. Eucleidean distances are used for this
    algorithm. The first point selected is based on the starting_idx provided and becomes the first
    query point. An approximation of the furthest point to the query point is found using
    find_furthest_neighbor and is selected. find_nearest_neighbor is then done to eliminate close
    neighbors to the new selected point. Medoid is then calculated from previously selected points
    and is used as the new query point for find_furthest_neighbor, repeating the process. Terminates
    upon selecting requested number of points or if all available points exhausted.

    Adapted from: https://en.wikipedia.org/wiki/K-d_tree#Construction
    """

    def __init__(self,
                 start_id=0,
                 func_distance=lambda x, y: spatial.minkowski_distance(x, y) ** 2,
                 scaling=10,
                 ):
        """
        Initializing class.

        Parameters
        ----------
        start_id: int
            Index for the first point to be selected.
        func_distance: callable
            Function for calculating the pairwise distance between instances of the array.
        scaling: float
            Percent of average maximum distance to use when eliminating the closest points.
        """

        self.starting_idx = start_id
        self.func_distance = func_distance
        self.BT = collections.namedtuple("BT", ["value", "index", "left", "right"])
        self.FNRecord = collections.namedtuple("FNRecord", ["point", "index", "distance"])
        self.scaling = scaling / 100
        self.ratio = None

    def _kdtree(self, arr):
        """Construct a k-d tree from an iterable of points.

        Parameters
        ----------
        arr: list or np.ndarray
            Coordinate array of points.

        Returns
        -------
        kdtree: collections.namedtuple
            KDTree organizing coordinates.
        """

        k = len(arr[0])

        def build(points, depth, old_indices=None):
            """Build a k-d tree from a set of points at a given depth."""
            if len(points) == 0:
                return None
            middle = len(points) // 2
            indices, points = zip(*sorted(enumerate(points), key=lambda x: x[1][depth % k]))
            if old_indices is not None:
                indices = [old_indices[i] for i in indices]
            return self.BT(
                value=points[middle],
                index=indices[middle],
                left=build(
                    points=points[:middle],
                    depth=depth + 1,
                    old_indices=indices[:middle],
                ),
                right=build(
                    points=points[middle + 1:],
                    depth=depth + 1,
                    old_indices=indices[middle + 1:],
                ),
            )

        kdtree = build(points=arr, depth=0)
        return kdtree

    def _eliminate(self, tree, point, threshold, num_eliminate, bv):
        """Eliminates points from being selected in future rounds.

        Parameters
        ----------
        tree: spatial.KDTree
            KDTree organizing coordinates.
        point: list
            Point where close neighbors should be eliminated.
        threshold: float
            An average of all the furthest distances found using find_furthest_neighbor
        num_eliminate: int
            Maximum number of points permitted to be eliminated.
        bv: bitarray
            Bitvector marking picked/eliminated points.

        Returns
        -------
        num_eliminate: int
            Maximum number of points permitted to be eliminated.
        """
        _, elim_candidates = tree.query(point, k=self.ratio,
                                        distance_upper_bound=np.sqrt(threshold),
                                        workers=-1)
        if num_eliminate < 0:
            elim_candidates = elim_candidates[:num_eliminate]
        for index in elim_candidates:
            try:
                bv[index] = 1
                num_eliminate -= 1
            except IndexError:
                break
        return num_eliminate

    def _find_furthest_neighbor(self, kdtree, point, selected_bitvector):
        """Find approximately the furthest neighbor in a k-d tree for a given point.

        Parameters
        ----------
        kdtree: collections.namedtuple
            KDTree organizing coordinates.
        point: list
            Query point for search.
        selected_bitvector: bitarray
            Bitvector to keep track of previously selected points from array.

        Returns
        -------
        best: collections.namedtuple
            The furthest point found in search.
        """

        k = len(point)
        best = None

        def search(tree, depth):
            # Recursively search through the k-d tree to find the
            # furthest neighbor.

            nonlocal selected_bitvector
            nonlocal best

            if tree is None:
                return

            if not selected_bitvector[tree.index]:
                distance = self.func_distance(tree.value, point)
                if best is None or distance > best.distance:
                    best = self.FNRecord(point=tree.value, index=tree.index, distance=distance)

            axis = depth % k
            diff = point[axis] - tree.value[axis]
            if diff <= 0:
                close, away = tree.left, tree.right
            else:
                close, away = tree.right, tree.left

            search(tree=away, depth=depth + 1)
            if best is None or (close is not None and diff ** 2 <= 1.1 * (
                    (point[axis] - close.value[axis]) ** 2)):
                search(tree=close, depth=depth + 1)

        search(tree=kdtree, depth=0)
        return best

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """Main function for selecting points using the KDTree algorithm.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points
        num_selected: int
            Number of molecules that need to be selected.
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if cluster_ids is not None:
            arr = arr[cluster_ids]

        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        arr_len = len(arr)
        fartree = self._kdtree(arr)
        neartree = spatial.KDTree(arr)

        bv = bitarray.bitarray(arr_len)
        bv[:] = 0
        selected = [self.starting_idx]
        query_point = arr[self.starting_idx]
        bv[self.starting_idx] = 1
        count = 1
        num_eliminate = arr_len - num_selected
        self.ratio = math.ceil(num_eliminate / num_selected)
        best_distance_av = 0
        while len(selected) < num_selected:
            new_point = self._find_furthest_neighbor(fartree, query_point, bv)
            if new_point is None:
                return selected
            selected.append(new_point.index)
            bv[new_point.index] = 1
            query_point = (count * np.array(query_point) + np.array(new_point.point)) / (count + 1)
            query_point = query_point.tolist()
            if count == 1:
                best_distance_av = new_point.distance
            else:
                best_distance_av = (count * best_distance_av + new_point.distance) / (count + 1)
            if count == 1:
                if num_eliminate > 0 and self.scaling != 0:
                    num_eliminate = self._eliminate(neartree, arr[self.starting_idx],
                                                    best_distance_av * self.scaling,
                                                    num_eliminate, bv)
            if num_eliminate > 0 and self.scaling != 0:
                num_eliminate = self._eliminate(neartree, new_point.point,
                                                best_distance_av * self.scaling,
                                                num_eliminate, bv)
            count += 1
        return selected
