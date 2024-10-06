# -*- coding: utf-8 -*-
#
# The Selector is a Python library of algorithms for selecting diverse
# subsets of data for machine-learning.
#
# Copyright (C) 2022-2024 The QC-Devs Community
#
# This file is part of Selector.
#
# Selector is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Selector is distributed in the hope that it will be useful,
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
import numpy as np
import scipy

from selector.measures.diversity import compute_diversity
from selector.methods.base import SelectionBase

__all__ = [
    "GridPartition",
    "Medoid",
]


class GridPartition(SelectionBase):
    r"""Select subset of sample points using the grid partitioning algorithms.

    Given the number of bins along each axis, samples are partitioned using various methods [1]_:

    - The `equisized_independent` partitions the feature space into bins of equal size along
      each dimension.

    - The `equisized_dependent` partitions the space where the bins can have different length
      in each dimension. I.e., the `l-`th dimension bins depend on the previous dimensions.
      So, the order of features affects the outcome.

    - The `equifrequent_independent` divides the space into bins with approximately equal
      number of sample points in each bin.

    - The `equifrequent_dependent` is similar to `equisized_dependent` where the partition in
      each dimension will depend on the previous dimensions.

    References
    ----------
    .. [1] Bayley, Martin J., and Peter Willett. "Binning schemes for partition-based
           compound selection." Journal of Molecular Graphics and Modelling 17.1 (1999): 10-18.
    """

    def __init__(
        self, nbins_axis: int, bin_method: str = "equisized_independent", random_seed: int = 42
    ):
        """Initialize class.

        Parameters
        ----------
        nbins_axis: int
            Number of bins to partition each axis into. The total number of resulting bins is
            `numb_bins_axis` raised to the power of the dimensionality of the feature space.
        bin_method: str, optional
            Method used to partition the sample points into bins. Options include:
            "equisized_independent", "equisized_dependent", "equifrequent_independent" and
            "equifrequent_dependent".
        random_seed: int, optional
            Seed for random selection of sample points from each bin.
        """
        if not isinstance(nbins_axis, int):
            raise TypeError(f"Number of bins should be integer, got {type(nbins_axis)}.")
        if not isinstance(random_seed, int):
            raise TypeError(f"The random seed should be integer, got {type(random_seed)}.")
        if not isinstance(bin_method, str):
            raise TypeError(f"The bin_method should be a string, got {type(bin_method)}.")
        self.random_seed = random_seed
        self.nbins_axis = nbins_axis
        self.bin_method = bin_method

    @staticmethod
    def partition_points_to_bins_equisized(X, nbins_axis):
        r"""
        Find all bins ids that has points in them and assign each point to each of those bins.

        For each `n_features` dimensions, get the minimum and maximum of feature to and use
        `nbins_axis` to compute length of the bin. Then assign sample points to bins along
        each axis/dimension of feature space.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Feature matrix of `n_samples` samples in `n_features` dimensional space.
        nbins_axis: int
            Number of bins along each axis/dimension of feature space.

        Returns
        -------
        unique_bin_indices: ndarray(int,)
            Unique (without duplication) bin indices that have at least one sample point.
            These are integer tuples :math:`(i_1, \cdot, i_\text{n_features})` with elements
            corresponding to the bin index along each axis/dimension of feature space.
            `inverse_ids` contains indices of `unique_bins_ids` for each of the :math:`N` points that
            it is assigned to.
        inverse_indices: ndarray(int,)
            Indices of the unique bins (along specified axis) that can be used to reconstruct bin
            index of each sample (`unique_bin_indices[inverse_indices]` gives bin index array).
        """
        # find the minimum and maximum of features along axis/dimension
        axis_minimum = np.min(X, axis=0)
        axis_maximum = np.max(X, axis=0)
        bin_length = (axis_maximum - axis_minimum) / nbins_axis
        # assign each sample to a bin along each dimension (floor_divide returns array of integers)
        bin_index = np.floor_divide(X - axis_minimum, bin_length)
        # get unique bin indices (occupied by samples) and indices of the unique array
        # (along specified axis) that can be used to reconstruct bin_index
        # in other words, unique_bin_index[inverse_index] gives back bin_index array
        unique_bin_index, inverse_index = np.unique(bin_index, return_inverse=True, axis=0)
        return unique_bin_index, inverse_index

    @staticmethod
    def partition_points_to_bins_equifrequent(X, nbins_axis):
        r"""
        Find all bins ids that contains points using the equifrequent method.

        The equifrequent method partitions each bin to have equal number of points.
        This is done by doing a linear interpolation from integer indices and points, where
        it is then evaluated on a uniform grid with number of bins as the spacing in each axis.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
           Feature matrix of `n_samples` samples in `n_features` dimensional space.
        nbins_axis: int
            Number of bins along each axis or feature dimension.

        Returns
        -------
        unique_bin_indices: ndarray(int,)
            Unique (without duplication) bin indices that have at least one sample point.
            These are integer tuples :math:`(i_1, \cdot, i_\text{n_features})` with elements
            corresponding to the bin index along each axis/dimension of feature space.
            `inverse_ids` contains indices of `unique_bins_ids` for each of the :math:`N` points that
            it is assigned to.
        inverse_indices: ndarray(int,)
            Indices of the unique bins (along specified axis) that can be used to reconstruct bin
            index of each sample (`unique_bin_indices[inverse_indices]` gives bin index array).
        """
        n_samples = len(X)
        # to obtain the lower and upper range of bins so that each bin has equal number of points,
        # we interpolate the feature value for indices delineating the range of bins.
        # I.e., sorting the features values along one axis, the (xp, fp) pairs are formed where
        # xp denote the integer index of `n_samples points and fp the corresponding feature values.
        # now, the value of feature is interpolated for the indices corresponding to lower and
        # upper ranges of bins denoted by x. The x indices are obtained by evenly dividing the
        # (0, n_samples) into nbins_axis bins (hence nbins_axis + 1 for number of samples indices).
        # Obviously, in this way, the bins have roughly `n_samples` points.
        # Note that the starting and ending indices of the bins are always 0 and n_samples,
        # corresponding to the minimum and maximum of the feature values.
        # The resulting bins_edge defines a monotonically increasing array of bin edges, including
        # the rightmost edge, allowing for non-uniform bin widths.
        # The bins_edge[0] and bins_edge[-1] correspond to the min and max of the feature values.
        bins_edge = np.interp(
            x=np.linspace(0, n_samples, nbins_axis + 1), xp=np.arange(n_samples), fp=np.sort(X)
        )
        # Note: alternatively, one can use x=np.linspace(0, n_samples - 1, nbins_axis + 1) so
        # that the ending index corresponds to the index of the last sample point. This would not
        # be an issue, because the numpy interpolate function np.interp has two attributes called
        # right/left, so if the value x is outside the interpolating domain, then it returns the
        # closest data point fp[-1]/fp[0], respectively. This causes the partition into bins to
        # always include the endpoints.

        # To assign samples into bins, sample features are subtracted from the bins_edge, and
        # the index of the bins_edge where the difference switches from negative to positive is
        # the bin index that the sample is assigned to. The switch from negative to positive is
        # identified by the argmax function after setting all the non-negative values to -inf.
        pt_to_bind = bins_edge - X[:, None]
        pt_to_bind[pt_to_bind >= 0.0] = -np.inf
        bin_index = np.argmax(pt_to_bind, axis=1)
        unique_bin_ids, inverse_ids = np.unique(bin_index, return_inverse=True)
        return unique_bin_ids, inverse_ids

    def get_bins_from_method(self, X):
        r"""Assign sample points to bins based on the partitioning method.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
           Feature matrix of `n_samples` samples in `n_features` dimensional space.

        Returns
        -------
        bins: dict[Tuple(int), List[int]]
            Dictionary of bins where keys are the unique bin indices (that contain at least one
            sample point) and the values are the list of sample indices in that bin.
        """

        # dictionary of bins where the keys are the unique bin indices (that contain at least one
        # sample) and the values are the list of sample indices in that bin.
        bins = {}

        if self.bin_method == "equisized_independent":
            # partition each dimension/feature independently into `num_bins_axis` bins
            unique_bin_index, inverse_index = self.partition_points_to_bins_equisized(
                X, self.nbins_axis
            )
            # populate bins dictionary
            for i, key in enumerate(unique_bin_index):
                bins[tuple(key)] = list(np.where(inverse_index == i)[0])

        elif self.bin_method == "equisized_dependent":
            # partition the first dimension (1st feature axis) into `num_bins_axis` bins
            unique_bin_index, inverse_index = self.partition_points_to_bins_equisized(
                X[:, 0], self.nbins_axis
            )
            # populate bins dictionary based on the 1st feature
            for i, key in enumerate(unique_bin_index):
                bins[tuple([key])] = list(np.where(inverse_index == i)[0])

            # loop over the remaining dimensions (2nd to last feature axis), and for each axis
            # partition the points in each bin of the previous axes into `num_bins_axis` bins
            # as a result, each iteration adds a new dimension to the bins dictionary
            for index_feature in range(1, X.shape[1]):
                # make a dictionary to store the bins for the current axis
                bins_axis = {}
                # divide points in each bin into `num_bins_axis` bins based on the i-th feature
                for bin, index_samples in bins.items():
                    # equisized partition of points in bin along i-th feature
                    unique_bin_index, inverse_index = self.partition_points_to_bins_equisized(
                        X[index_samples, index_feature], self.nbins_axis
                    )
                    # update the bins_axis to include the new dimension/feature for the current bin
                    for i, bin_index in enumerate(unique_bin_index):
                        # form a new bin_index by appending  current bin_index to the previous bin
                        key = tuple(list(bin) + [bin_index])
                        bins_axis.update(
                            {key: list(np.array(index_samples)[np.where(inverse_index == i)[0]])}
                        )
                bins = bins_axis

        elif self.bin_method == "equifrequent_independent":
            # partition each dimension of feature space independently into `num_bins_axis` bins
            bins_features = np.zeros(X.shape, dtype=int)
            for index_feature in range(0, X.shape[1]):
                unique_bin_index, inverse_index = self.partition_points_to_bins_equifrequent(
                    X[:, index_feature], self.nbins_axis
                )
                bins_features[:, index_feature] = unique_bin_index[inverse_index]
            unique_bin_index, inverse_index = np.unique(bins_features, return_inverse=True, axis=0)

            # populate bins dictionary
            for i, key in enumerate(unique_bin_index):
                bins[tuple(key)] = list(np.where(inverse_index == i)[0])

        elif self.bin_method == "equifrequent_dependent":
            # partition the first dimension (1st feature axis) into `num_bins_axis` bins
            unique_bin_index, inverse_index = self.partition_points_to_bins_equifrequent(
                X[:, 0], self.nbins_axis
            )
            # populate bins dictionary based on the 1st feature
            for i, key in enumerate(unique_bin_index):
                bins[tuple([key])] = list(np.where(inverse_index == i)[0])

            # loop over the remaining dimensions (2nd to last feature axis), and for each axis
            # partition the points in each bin of the previous axes into `num_bins_axis` bins
            for index_feature in range(1, X.shape[1]):
                # make a dictionary to store the bins for the current axis
                bins_axis = {}
                # divide points in each bin, based on the i-th feature, into `num_bins_axis` bins
                for bin, index_samples in bins.items():
                    # equifrequent partition of points in bin along i-th feature
                    unique_bin_index, inverse_index = self.partition_points_to_bins_equifrequent(
                        X[index_samples, index_feature], self.nbins_axis
                    )
                    # update the bins_axis to include the new dimension/feature for the current bin
                    for i, key in enumerate(unique_bin_index):
                        # form a new bin_index by appending  current bin_index to the previous bin
                        key = tuple(list(bin) + [key])
                        bins_axis.update(
                            {key: list(np.array(index_samples)[np.where(inverse_index == i)[0]])}
                        )
                bins = bins_axis
        else:
            raise ValueError(f"{self.bin_method} not a valid bin_method")

        return bins

    def select_from_bins(
        self,
        X,
        bins,
        num_selected,
        diversity_type="hypersphere_overlap",
        cs=None,
    ):
        r"""
        From the bins, select a certain number of points of the bins.

        Points are selected in an iterative manner. If the number of points needed to be selected
         is greater than number of bins left then randomly select points from each of the bins. If
         any of the bins are empty then remove the bins. If it is less than the number of bins left,
         then calculate the diversity of each bin and choose points of bins with the highest diversity.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Feature matrix of `n_samples` samples in `n_features` dimensional space.
        bins: dict(tuple(int), list[int])
            The bins that map to the id the bin (as a tuple of integers) and returns
            the indices of the points that are contained in that bin.
        num_selected: int
            Number of points to select from the bins.
        diversity_type: str, optional
            Type of diversity to use. Default="hypersphere_overlap".
        cs : int, optional
            Number of common substructures in molecular compound dataset. Used only if calculating
            `explicit_diversity_index`. Default is "None".

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
            # if the number of samples that should be selected is greater than number of bins,
            # randomly select points from the bins.
            if len(bins) <= num_needed:
                # Go through each bin and select a point at random from it and delete it later
                for bin_idx, bin_list in bins.items():
                    random_int = rng.integers(low=0, high=len(bin_list), size=1)[0]
                    sample_index = bin_list.pop(random_int)
                    selected.append(sample_index)
                    if len(bin_list) == 0:
                        to_delete.append(bin_idx)
                for idx in to_delete:
                    del bins[idx]
                to_delete = []
            else:
                # If number of samples that should be selected is less than the number of bins,
                # calculate the diversity of each bin and select samples from bins with highest
                # diversity.
                diversity = [
                    (
                        compute_diversity(
                            features=X,
                            feature_subset=X[bin_list, :],
                            div_type=diversity_type,
                            cs=cs,
                        ),
                        bin_idx,
                    )
                    for bin_idx, bin_list in bins.items()
                ]
                diversity.sort(reverse=True)
                for _, bin_idx in diversity[:num_needed]:
                    random_int = rng.integers(low=0, high=len(bins[bin_idx]), size=1)[0]
                    sample_index = bins[bin_idx].pop(random_int)
                    selected.append(sample_index)
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
            raise TypeError(
                f"cluster_ids {type(cluster_ids)} should be either None or numpy " f"array."
            )

        if cluster_ids is not None:
            X = X[cluster_ids]
        bins = self.get_bins_from_method(X)
        selected = self.select_from_bins(X, bins, num_selected)
        return selected


class Medoid(SelectionBase):
    """Selecting points using an algorithm adapted from KDTree.

    Points are initially used to construct a KDTree. Euclidean distances are used for this
    algorithm. The first point selected is based on the ref_index provided and becomes the first
    query point. An approximation of the furthest point to the query point is found using
    find_furthest_neighbor and is selected. find_nearest_neighbor is then done to eliminate close
    neighbors to the new selected point. Medoid is then calculated from previously selected points
    and is used as the new query point for find_furthest_neighbor, repeating the process. Terminates
    upon selecting requested number of points or if all available points exhausted.

    Adapted from: https://en.wikipedia.org/wiki/K-d_tree#Construction
    """

    def __init__(
        self,
        func_distance=lambda x, y: scipy.spatial.minkowski_distance(x, y) ** 2,
        ref_index=0,
        scaling=10,
    ):
        """
        Initializing class.

        Parameters
        ----------
        fun_distance : callable
            Function for calculating the pairwise distance between sample points.
            `fun_dist(X) -> X_dist` takes a 2D feature array of shape (n_samples, n_features)
            and returns a 2D distance array of shape (n_samples, n_samples).
        ref_index : int, optional
            Index for the sample to start selection from; this index is the first sample selected.
        scaling: float
            Percent of average maximum distance to use when eliminating the closest points.

        Notes
        -----
        The `Mediod` implementation is based on the KDTree algorithm and therefore can give
        different results for cases with duplicated points or the same features for different
        objects in the original feature space. This is dicussed in
        https://github.com/theochem/Selector/issues/238.
        This is because the same features lead to the same distances in the tree, and this is a
        known issue of sorting the points and indices in the KDTree algorithm, as discussed
        in https://github.com/scipy/scipy/issues/19029. Therefore, precautions should be taken if
        duplicated points are present in the dataset.

        """

        self.starting_idx = ref_index
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

            # sort the points and indices
            # indices, points = zip(*sorted(enumerate(points), key=lambda x: x[1][depth % k]))
            indices = np.argsort(np.array(points)[:, depth % k], kind="stable")
            points = np.array(points)[indices]

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
                    points=points[middle + 1 :],
                    depth=depth + 1,
                    old_indices=indices[middle + 1 :],
                ),
            )

        kdtree = build(points=arr, depth=0)
        return kdtree

    def _eliminate(self, tree, point, threshold, num_eliminate, bv):
        """Eliminates points from being selected in future rounds.

        Parameters
        ----------
        tree: scipy.spatial.KDTree
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
        # elim_candidates can be integer or array of integers
        # https://github.com/scipy/scipy/blob/a2a287d1f7c81154256ba742b4b8bb108a612166/scipy/spatial/_kdtree.py#L476
        if isinstance(elim_candidates, np.intp):
            elim_candidates = [elim_candidates]

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
        neartree = scipy.spatial.KDTree(arr)

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
