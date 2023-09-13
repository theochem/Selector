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
from DiverseSelector.diversity import hypersphere_overlap_of_subset
from DiverseSelector.methods.utils import optimize_radius
import numpy as np
from scipy import spatial


__all__ = [
    "DirectedSphereExclusion",
    "GridPartitioning",
    "Medoid",
]


class DirectedSphereExclusion(SelectionBase):
    """Select samples using Directed Sphere Exclusion (DISE) algorithm.

    In a nutshell, this algorithm iteratively excludes any sample within a given radius from
    any already selected sample. The radius of the exclusion sphere is an adjustable parameter.
    Compared to Sphere Exclusion algorithm, the Directed Sphere Exclusion algorithm achieves a
    more evenly distributed subset selection by abandoning the random selection approach and
    instead imposing a directed selection.

    Reference sample is chosen based on the `ref_index`, which is excluded from the selected
    subset. All samples are sorted (ascending order) based on their Minkowski p-norm distance
    from the reference sample. Looping through sorted samples, the sample is selected if it is
    not already excluded. If selected, all its neighboring samples within a sphere of radius r
    (i.e., exclusion sphere) are excluded from being selected. When the selected number of points
    is greater than specified subset `size`, the selection process terminates. The `r0` is used
    as the initial radius of exclusion sphere, however, it is optimized to select the desired
    number of samples.

    Notes
    -----
    Gobbi, A., and Lee, M.-L. (2002). DISE: directed sphere exclusion.
    Journal of Chemical Information and Computer Sciences,
    43(1), 317–323. https://doi.org/10.1021/ci025554v
    """

    def __init__(self, r0=None, ref_index=0, p=2.0, eps=0.0, tol=0.05, n_iter=10, random_seed=42):
        """Initialize class.

        Parameters
        ----------
        r0: float, optional
            Initial guess for radius of the exclusion sphere.
        ref_index: int, optional
            Index of the reference sample to start the selection algorithm from.
            This sample is not included in the selected subset.
        p: float, optional
            Which Minkowski p-norm to use. The values of `p` should be within [1, inf].
            A finite large p may cause a ValueError if overflow can occur.
        eps: float, optional
            Approximate nearest neighbor search used in `KDTree.query_ball_tree`.
            Branches of the tree are not explored if their nearest points are further than
            r/(1+eps), and branches are added in bulk if their furthest points are nearer than
            r * (1+eps). eps has to be non-negative.
        tol: float, optional
            Percentage error of number of samples actually selected from number of samples requested.
        n_iter: int, optional
            Number of iterations for optimizing the radius of exclusion sphere.
        random_seed: int, optional
            Seed for random selection of points be evaluated.
        """
        self.r = r0
        self.ref_index = ref_index
        self.p = p
        self.eps = eps
        self.tol = tol
        self.n_iter = n_iter
        self.random_seed = random_seed

    def algorithm(self, X, max_size):
        """Return selected samples based on directed sphere exclusion algorithm.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
           Feature matrix of `n_samples` samples in `n_features` dimensional space.
        max_size: int
            Maximum number of samples to select.

        Returns
        -------
        selected: list
            List of indices of selected samples.
        """

        # calculate distance of all samples from reference sample; distance is a (n_samples,) array
        distances = scipy.spatial.minkowski_distance(X[self.ref_index], X, p=self.p)
        # get sorted index of samples based on their distance from reference (closest to farthest)
        index_sorted = np.argsort(distances)
        # construct KDTree for quick nearest-neighbor lookup
        kdtree = spatial.KDTree(X)

        # construct bitarray to track selected samples (1 means exclude)
        bv = bitarray.bitarray(list(np.zeros(len(X), dtype=int)))
        bv[self.ref_index] = 1

        selected = []
        for idx in index_sorted:
            # select sample if it is not already excluded from consideration
            # indexing a single item of a bitarray will always return an integer
            if bv[idx] == 0:
                selected.append(idx)
                # return indices of selected samples, if desired number is selected
                if len(selected) > max_size:
                    return selected
                # find index of all samples within radius of sample idx (this includes the sample index itself)
                index_exclude = kdtree.query_ball_point(
                    X[idx], self.r, eps=self.eps, p=self.p, workers=-1
                )
                # exclude samples within radius r of sample idx (measure by Minkowski p-norm) from
                # future consideration by setting their bitarray value to 1
                for index in index_exclude:
                    bv[index] = 1

        return selected

    def select_from_cluster(self, X, size, cluster_ids=None):
        """Return selected samples from a cluster based on directed sphere exclusion algorithm

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
           Feature matrix of `n_samples` samples in `n_features` dimensional space.
        size: int
            Number of samples to be selected.
        cluster_ids: np.ndarray
            Indices of samples that form a cluster.

        Returns
        -------
        selected: list
            List of indices of selected samples.
        """
        if X.shape[0] < size:
            raise RuntimeError(
                f"Number of samples is less than the requested sample size: {X.shape[0]} < {size}."
            )
        return optimize_radius(self, X, size, cluster_ids)


class GridPartitioning(SelectionBase):
    r"""Selecting points using the Grid Partitioning algorithm.

    Points are partitioned into grids using various methods.

    - The equisized independent partitions the space into bins of equal size in all dimensions.
      This is determined by the user based on the number of bins to have in each
      axis/dimension.

    - The equisized dependent partitions space where the bins can be of
      different length in each dimension, each `l`th dimension partition depends on the previous
      `l-1`th dimension.  The order of points affects the latter method.

    - The equifrequent independent partitions the space into bins of approximately equal
      number of points in each bins.

    - The equifrequent dependent is similar to equisized dependent where the partition in each
      dimension will depend on the previous dimensions.

    References
    ----------
    .. [1] Bayley, Martin J., and Peter Willett. "Binning schemes for partition-based
           compound selection." Journal of Molecular Graphics and Modelling 17.1 (1999): 10-18.
    """

    def __init__(
        self,
        numb_bins_axis: int,
        grid_method: str = "equisized_independent",
        random_seed: int = 42
    ):
        """
        Initializing class.

        Parameters
        ----------
        numb_bins_axis: int
            Number of bins/cells to partition each axis into, the number of resulting grids is 
            bins/cells to the power of the dimensionality of the coordinate array.
        grid_method: str, optional
            Grid method used to partition the points into grids. "equisized_independent",
            "equisized_dependent", "equifrequent_independent" and "equifrequent_dependent"
            are supported options.
        random_seed: int, optional
            Seed for random selection of points to be selected from each grid.
        """
        if not isinstance(numb_bins_axis, int):
            raise TypeError(f"Number of bins {type(numb_bins_axis)} should be integer.")
        if not isinstance(random_seed, int):
            raise TypeError(f"The random seed {type(random_seed)} should be integer.")
        if not isinstance(grid_method, str):
            raise TypeError(f"The grid method {type(grid_method)} should be a string.")
        self.random_seed = random_seed
        self.numb_bins_axis = numb_bins_axis
        self.grid_method = grid_method

    def partition_points_to_bins_equisized(self, features):
        r"""
        Find all bins ids that has points in them and assign each point to each of those bins.

        Go through each :math:`D` dimension and get the minimum, maximum and length of each cell
           of that dimension. This is then used to define the size of the bins, where the bins are
           equal length within each dimension.

        Parameters
        ----------
        features: ndarray(N, D)
            The points used to partition the grid into bins and assign each points to each bin.

        Returns
        -------
        unique_bins_ids, inverse_ids: ndarray, ndarray(int,)
            `unique_bins_ids` is the unique (without duplication) bin ids that points are assigned to.
            The bin ids are integer arrays :math:`(i_1, \cdot, i_D)` that corresponds to each partition
            in each dimension.
            `inverse_ids` contains indices of `unique_bins_ids` for each of the :math:`N` points that
            it is assigned to.
        """
        axis_minimum = np.min(features, axis=0)
        axis_maximum = np.max(features, axis=0)
        cell_length = (axis_maximum - axis_minimum) / self.numb_bins_axis
        # Rows correspond to the points and the columns correspond to bin id
        bin_ids = np.array(np.floor_divide(features - axis_minimum, cell_length), dtype=int)
        # `unique_bin_ids` corresponds to unique bin ids and `inverse_ids` which point corresponds
        #   to which bin
        unique_bin_ids, inverse_ids = np.unique(bin_ids, return_inverse=True, axis=0)
        return unique_bin_ids, inverse_ids

    def partition_points_to_bins_equifrequent(self, features):
        r"""
        Find all bins ids that contains points using the equifrequent method.

        The equifrequent method partitions each bin to have equal number of points.
        This is done by doing a linear interpolation from integer indices and points, where
        it is then evaluated on a uniform grid with number of bins as the spacing in each axis.

        Parameters
        ----------
        features: ndarray(N, D)
            The points used to partition the grid into bins and assign each points to each bin.

        Returns
        -------
        unique_bins_ids, inverse_ids: ndarray, ndarray(int,)
            `unique_bins_ids` is the unique (without duplication) bin ids that points are assigned to.
            The bin ids are integer arrays :math:`(i_1, \cdot, i_D)` that corresponds to each partition
            in each dimension.
            `inverse_ids` contains indices of `unique_bins_ids` for each of the :math:`N` points that
            it is assigned to.
        """
        n_pt = len(features)

        # This gives [l_1, l_2, l_3, ..., l_{B+1}] where [l_1, l_2] is the first bin and
        #    [l_2, l_3] is the second bin. Note that l_1 is always the minimum and l_{B+1}
        #    is always the maximum.  interp does linear interpolation between integer indices
        #    and the output is the ith dimension of the points.
        bins_length = np.interp(np.linspace(0, n_pt, self.numb_bins_axis + 1),
                                np.arange(n_pt),
                                np.sort(features))
        # Using the bin length, partition each point to the correct bin. This is done by
        #   subtracting each point with each interval \{l_i\} and where it switches
        #   from negative to positive, is the bin index that it is assigned to.
        pt_to_bind = bins_length - features[:, None]
        pt_to_bind[pt_to_bind >= 0.0] = -np.inf
        bin_ids_oned = np.argmax(pt_to_bind, axis=1)
        unique_bin_ids, inverse_ids = np.unique(bin_ids_oned, return_inverse=True)
        return unique_bin_ids, inverse_ids

    def get_bins_from_method(self, X):
        r"""
        From the partitioning method, obtain the bins ids and points contained in each bin.

        Parameters
        ----------
        X: ndarray(N, D)
            The points used to partition the grid into bins and assign each points to each bin.

        Returns
        -------
        bins: dict[Tuple(int), List[int]]
            Dictionary whose keys are the ids of each of the bins and whose items are
            a list of integers specifying the point belongs to that bin.
        """
        data_dim = X.shape[1]
        if self.grid_method == "equisized_independent":
            unique_bin_ids, inverse_ids = self.partition_points_to_bins_equisized(X)
            # The unique bins ids is the keys and the items are the list of indices of points that
            #   corresponds to that bin.
            bins = {
                tuple(key): list(np.where(inverse_ids == i)[0])
                for i, key in enumerate(unique_bin_ids)
            }
        elif self.grid_method == "equisized_dependent":
            # Partition the first dimension using the same procedure as `equisized_independent`.
            unique_bin_ids, inverse_ids = self.partition_points_to_bins_equisized(X[:, 0])
            bins = {
                tuple([key, ]): list(np.where(inverse_ids == i)[0])
                for i, key in enumerate(unique_bin_ids)
            }
            # Apply the partition scheme at the next features/dimension dependent on the previous
            #   partition
            for i in range(1, data_dim):
                new_bins = {}  # Need another dictionary because bins is being iterated on
                for bin_idx, bin_list in bins.items():
                    # Go through each bin and grab the new axis based on it and partition as usual
                    unique_bin_ids, inverse_ids = self.partition_points_to_bins_equisized(X[bin_list, i])
                    # Add the previous bin_ids to the current unique_bin_ids.
                    unique_bin_ids = np.array([list(bin_idx) + [x] for x in unique_bin_ids])
                    # Update the new bins to include the next dimension/feature.
                    new_bins.update(
                        {
                            tuple(key): list(np.array(bin_list)[np.where(inverse_ids == i)[0]])
                            for i, key in enumerate(unique_bin_ids)
                        }
                    )
                bins = new_bins
        elif self.grid_method == "equifrequent_independent":
            npt, ndim = X.shape
            all_bins = np.zeros(X.shape, dtype=int)
            for i in range(0, ndim):
                # This gives [l_1, l_2, l_3, ..., l_{B+1}] where [l_1, l_2] is the first bin and
                #    [l_2, l_3] is the second bin. Note that l_1 is always the minimum and l_{B+1}
                #    is always the maximum.  interp does linear interpolation between integer indices
                #    and the output is the ith dimension of the points.
                bins_length = np.interp(np.linspace(0, npt, self.numb_bins_axis + 1),
                                        np.arange(npt),
                                        np.sort(X[:, i]))

                # Using the bin length, partition each point to the correct bin. This is done by
                #   subtracting each point with each interval \{l_i\} and where it switches
                #   from negative to positive, is the bin index that it is assigned to.
                pt_to_bind = bins_length - X[:, i][:, None]
                pt_to_bind[pt_to_bind >= 0.0] = -np.inf
                all_bins[:, i] = np.argmax(pt_to_bind, axis=1)

            unique_bin_ids, inverse_ids = np.unique(all_bins, return_inverse=True, axis=0)

            # The unique bins ids is the keys and the items are the list of indices of points that
            #   corresponds to that bin.
            bins = {
                tuple(key): list(np.where(inverse_ids == i)[0])
                for i, key in enumerate(unique_bin_ids)
            }

        elif self.grid_method == "equifrequent_dependent":
            # Partition the first using the same procedure as `equifrequent_independent`.
            unique_bin_ids, inverse_ids = self.partition_points_to_bins_equifrequent(X[:, 0])
            # The unique bins ids is the keys and the items are the list of indices of points that
            #   corresponds to that bin.
            bins = {
                tuple([key, ]): list(np.where(inverse_ids == i)[0]) for i, key in enumerate(unique_bin_ids)
            }
            # Do the next following dimensions
            for i in range(1, X.shape[1]):
                new_bins = {}  # Need another dictionary because bins is being iterated on
                for bin_idx, bin_list in bins.items():
                    pts_in_bin = X[bin_list, i]
                    unique_bin_ids, inverse_ids = self.partition_points_to_bins_equifrequent(pts_in_bin)
                    # Add the previous bin_ids to the current unique_bin_ids.
                    unique_bin_ids = np.array([list(bin_idx) + [x] for x in unique_bin_ids])
                    # Update the new bins to include the next dimension/feature.
                    new_bins.update(
                        {
                            tuple(key): list(np.array(bin_list)[np.where(inverse_ids == i)[0]])
                            for i, key in enumerate(unique_bin_ids)
                        }
                    )
                bins = new_bins
        else:
            raise ValueError(f"{self.grid_method} not a valid grid_method")

        return bins

    def select_from_bins(self, X, bins, num_selected):
        r"""
        From the bins, select a certain number of points of the bins.

        Points are selected in an iterative manner. If the number of point needed to be selected
         is greater than number of bins left then randomly select points from each of the bins. If
         any of the bins are empty then remove the binds. If it is less than the number of bins left,
         then calculate the diversity of each bin and choose points of bins with the highest diversity.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Feature matrix of `n_samples` samples in `n_features` dimensional space.
        bins: dict(tuple(int), list[int])
            The bins that map to the id the bin (as a typle of integers) and returns
            the indices of the points that are contained in that bin.
        num_selected: int
            Number of points to select from the bins.

        Returns
        -------
        List[int]:
            Indices of the points that were selected.
        """
        old_len = 0
        to_delete = []
        selected = []
        rng = np.random.default_rng(seed=self.random_seed)
        while len(selected) < num_selected:
            num_needed = num_selected - len(selected)
            bin_count = len(bins)
            # If the number of point needed is greater than number of bins then randomly
            #   select points from the bins.
            if bin_count <= num_needed:
                # Go through each bin and select a point at random from it and delete it later
                print(bin_count, num_needed)
                for bin_idx, bin_list in bins.items():
                    random_int = rng.integers(low=0, high=len(bin_list), size=1)[0]
                    mol_id = bin_list.pop(random_int)
                    selected.append(mol_id)
                    if len(bin_list) == 0:
                        to_delete.append(bin_idx)
                for idx in to_delete:
                    del bins[idx]
                print(bin_count, num_needed)
                to_delete = []
            else:
                # If number of points is less than the number of bins,
                # Calculate the diversity of each bin and pick based on the highest diversity
                diversity = [
                    (hypersphere_overlap_of_subset(X, X[bin_list, :]), bin_idx) for bin_idx, bin_list in bins.items()
                ]
                diversity.sort(reverse=True)
                for _, bin_idx in diversity[:num_needed]:
                    random_int = rng.integers(low=0, high=len(bins[bin_idx]), size=1)[0]
                    mol_id = bins[bin_idx].pop(random_int)
                    selected.append(mol_id)
            if len(selected) == old_len:
                break
            old_len = len(selected)
        return selected

    def select_from_cluster(self, X: np.ndarray, num_selected: int, cluster_ids: np.ndarray = None):
        """
        Grid partitioning algorithm for selecting points from cluster.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Feature matrix of `n_samples` samples in `n_features` dimensional space.
        num_selected: int
            Number of molecules that need to be selected.
        cluster_ids: ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list[int]
            List of ids of selected molecules with size `num_selected`.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X {type(X)} should of type numpy array.")
        if not isinstance(num_selected, int):
            raise TypeError(f"num_selected {type(num_selected)} should be of type int.")
        if cluster_ids is not None and not isinstance(cluster_ids, np.ndarray):
            raise TypeError(f"cluster_ids {type(cluster_ids)} should be either None or numpy "
                            f"array.")

        if cluster_ids is not None:
            X = X[cluster_ids]
        bins = self.get_bins_from_method(X)
        selected = self.select_from_bins(X, bins, num_selected)
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

    def __init__(
        self,
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
        _, elim_candidates = tree.query(
            point, k=self.ratio, distance_upper_bound=np.sqrt(threshold), workers=-1
        )
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
            if best is None or (
                close is not None and diff**2 <= 1.1 * ((point[axis] - close.value[axis]) ** 2)
            ):
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
                    num_eliminate = self._eliminate(
                        neartree,
                        arr[self.starting_idx],
                        best_distance_av * self.scaling,
                        num_eliminate,
                        bv,
                    )
            if num_eliminate > 0 and self.scaling != 0:
                num_eliminate = self._eliminate(
                    neartree, new_point.point, best_distance_av * self.scaling, num_eliminate, bv
                )
            count += 1
        return selected
