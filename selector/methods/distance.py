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
"""Module for Distance-Based Selection Methods."""

import bitarray
import numpy as np
from scipy import spatial
from typing import List, Iterable, Union

from selector.methods.base import SelectionBase
from selector.methods.utils import optimize_radius

__all__ = [
    "MaxMin",
    "MaxSum",
    "OptiSim",
    "DISE",
]


class MaxMin(SelectionBase):
    """Select samples using MaxMin algorithm.

    MaxMin is possibly the most widely used method for dissimilarity-based
    compound selection. When presented with a dataset of samples, the
    initial point is chosen as the dataset's medoid center. Next, the second
    point is chosen to be that which is furthest from this initial point.
    Subsequently, all following points are selected via the following
    logic:

    1. Find the minimum distance from every point to the already-selected ones.
    2. Select the point which has the maximum distance among those calculated
       in the previous step.

    In the current implementation, this method requires or computes the full pairwise-distance
    matrix, so it is not recommended for large datasets.

    References
    ----------
    [1] Ashton, Mark, et al., Identification of diverse database subsets using
    property‐based and fragment‐based molecular descriptions, Quantitative
    Structure‐Activity Relationships 21.6 (2002): 598-604.
    """

    def __init__(self, fun_dist=None, ref_index=None):
        """
        Initializing class.

        Parameters
        ----------
        fun_distance : callable
            Function for calculating the pairwise distance between sample points.
            `fun_dist(x) -> x_dist` takes a 2D feature array of shape (n_samples, n_features)
            and returns a 2D distance array of shape (n_samples, n_samples).
        ref_index: int, list, optional
            Index of the reference sample to start the selection algorithm from.
            It can be an integer, or a list of integers or None. When None, the medoid center is chosen as the reference
            sample.
            When the `ref_index` is a list for multiple classes, it will be shared among all clusters.
            If we want to use different reference indices for each class, we can perform the subset
            selection for each class separately where different `ref_index` parameters can be used.
            For example, if we have two classes, we can pass `ref_index=[0, 1]` to select samples
            from class 0 and `ref_index=[3, 6]` class 1 respectively.

        """
        self.fun_dist = fun_dist
        self.ref_index = ref_index

    def select_from_cluster(self, x, size, labels=None) -> Union[List, Iterable]:
        """Return selected samples from a cluster based on MaxMin algorithm.

        Parameters
        ----------
        x: ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Feature matrix of `n_samples` samples in `n_features` dimensional feature space,
            or the pairwise distance matrix between `n_samples` samples.
            If `fun_dist` is `None`, the `x` is assumed to be a square pairwise distance matrix.
        size: int
            Number of sample points to select (i.e. size of the subset).
        labels: np.ndarray
            Indices of samples that form a cluster.

        Returns
        -------
        selected : Union[List, Iterable]
            List of indices of selected samples.
        """
        # calculate pairwise distance between points
        x_dist = x
        if self.fun_dist is not None:
            x_dist = self.fun_dist(x)
        # check x_dist is a square symmetric matrix
        if x_dist.shape[0] != x_dist.shape[1]:
            raise ValueError(f"The pairwise distance matrix must be square, got {x_dist.shape}.")
        if np.max(abs(x_dist - x_dist.T)) > 1e-8:
            raise ValueError("The pairwise distance matrix must be symmetric.")

        if labels is not None:
            # extract pairwise distances from full pairwise distance matrix to obtain a new matrix
            # that only contains pairwise distances between samples within a given cluster
            x_dist = x_dist[labels][:, labels]

        # choosing initial point
        selected = get_initial_selection(
            x=None, x_dist=x_dist, ref_index=self.ref_index, fun_dist=None
        )

        # select following points until desired number of points have been obtained
        while len(selected) < size:
            # determine the min pairwise distances between the selected points and all other points
            min_distances = np.min(x_dist[selected], axis=0)
            # determine which point affords the maximum distance among the minimum distances
            # captured in min_distances
            new_id = np.argmax(min_distances)
            selected.append(new_id)

        selected = [int(i) for i in selected]

        return selected


class MaxSum(SelectionBase):
    """Select samples using MaxSum algorithm.

    Whereas the goal of the MaxMin algorithm is to maximize the minimum distance
    between any pair of distinct elements in the selected subset of a dataset,
    the MaxSum algorithm aims to maximize the sum of distances between all
    pairs of elements in the selected subset. When presented with a dataset of
    samples, the initial point is chosen as the dataset's medoid center. Next,
    the second point is chosen to be that which is furthest from this initial
    point. Subsequently, all following points are selected via the following
    logic:

    1. Determine the sum of distances from every point to the already-selected ones.
    2. Select the point which has the maximum sum of distances among those calculated
       in the previous step.

    References
    ----------
    [1] Borodin, Allan, Hyun Chul Lee, and Yuli Ye, Max-sum diversification, monotone
    submodular functions and dynamic updates, Proceedings of the 31st ACM SIGMOD-SIGACT-SIGAI
    symposium on Principles of Database Systems. 2012.
    """

    def __init__(self, fun_dist=None, ref_index=None):
        """
        Initializing class.

        Parameters
        ----------
        fun_dist : callable
            Function for calculating the pairwise distance between sample points.
            `fun_dist(x) -> x_dist` takes a 2D feature array of shape (n_samples, n_features)
            and returns a 2D distance array of shape (n_samples, n_samples).
        ref_index: int, list, optional
            Index of the reference sample to start the selection algorithm from.
            It can be an integer, or a list of integers or None. When None, the medoid center is chosen as the reference
            sample.
            When the `ref_index` is a list for multiple classes, it will be shared among all clusters.
            If we want to use different reference indices for each class, we can perform the subset
            selection for each class separately where different `ref_index` parameters can be used.
            For example, if we have two classes, we can pass `ref_index=[0, 1]` to select samples
            from class 0 and `ref_index=[3, 6]` class 1 respectively.

        """
        self.fun_dist = fun_dist
        self.ref_index = ref_index

    def select_from_cluster(self, x, size, labels=None) -> Union[List, Iterable]:
        """Return selected samples from a cluster based on MaxSum algorithm.

        Parameters
        ----------
        x: ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Feature matrix of `n_samples` samples in `n_features` dimensional feature space,
            or the pairwise distance matrix between `n_samples` samples.
            If `fun_dist` is `None`, the `x` is assumed to be a square pairwise distance matrix.
        size: int
            Number of sample points to select (i.e. size of the subset).
        labels: np.ndarray
            Indices of samples that form a cluster.

        Returns
        -------
        selected : Union[List, Iterable]
            List of indices of selected samples.

        """
        # calculate pairwise distance between points
        x_dist = x
        if self.fun_dist is not None:
            x_dist = self.fun_dist(x)
        # check x_dist is a square symmetric matrix
        if x_dist.shape[0] != x_dist.shape[1]:
            raise ValueError(f"The pairwise distance matrix must be square, got {x_dist.shape}.")
        if np.max(abs(x_dist - x_dist.T)) > 1e-8:
            raise ValueError("The pairwise distance matrix must be symmetric.")

        if labels is not None:
            # extract pairwise distances from full pairwise distance matrix to obtain a new matrix
            # that only contains pairwise distances between samples within a given cluster.
            x_dist = x_dist[labels][:, labels]

        # setting up initial point
        selected = get_initial_selection(
            x=None, x_dist=x_dist, ref_index=self.ref_index, fun_dist=None
        )
        # select following points until desired number of points have been obtained
        while len(selected) < size:
            # determine sum of pairwise distances between selected points and all other points
            sum_distances = np.sum(x_dist[selected], axis=0)
            # determine which point has the max sum of pairwise distances to already-selected points
            new_id = np.argmax(sum_distances)
            # make sure that new_id corresponds to a new point
            while new_id in selected:
                # set the sum of distances for the current point corresponding to new_id to 0
                sum_distances[new_id] = 0
                # find a different point with the maximum sum of pairwise distances to
                # already-selected points
                new_id = np.argmax(sum_distances)
            selected.append(new_id)

        selected = [int(i) for i in selected]
        return selected


class OptiSim(SelectionBase):
    """Selecting samples using OptiSim algorithm.

    The OptiSim algorithm selects samples from a dataset by first choosing the medoid center as the
    initial point. Next, points are randomly chosen and added to a subsample if they exist
    outside of radius r from all previously selected points (otherwise, they are discarded). Once k
    number of points have been added to the subsample, the point with the greatest minimum distance
    to the previously selected points is chosen. Then, the subsample is cleared and the process is
    repeated.

    Notes
    -----
    When the `ref_index` is a list for multiple classes, it will be shared among all clusters.
    If we want to use different reference indices for each class, we can perform the subset
    selection for each class separately where different `ref_index` parameters can be used.
    For example, if we have two classes, we can pass `ref_index=[0, 1]` to select samples from
    class 0 and `ref_index=[3, 6]` class 1 respectively.

    References
    ----------
    [1] J. Chem. Inf. Comput. Sci. 1997, 37, 6, 1181–1188. https://doi.org/10.1021/ci970282v

    """

    def __init__(
        self,
        r0=None,
        k=10,
        tol=0.01,
        n_iter=10,
        eps=0,
        p=2,
        random_seed=42,
        ref_index=0,
        fun_dist=None,
    ):
        """
        Initialize class.

        Parameters
        ----------
        r0 : float, optional
            Initial guess of radius for OptiSim algorithm. No points within this distance of an
            already selected point can be selected. If `None`, the maximum range of features and
            the size of subset are used to calculate the initial radius. This radius is optimized
            to result in the desired number of samples selected, if possible.
        k : int, optional
            Amount of points to add to subsample before selecting one of the points with the
            greatest minimum distance to the previously selected points.
        tol : float, optional
            Percentage error of number of samples actually selected from number of samples
            requested.
        n_iter : int, optional
            Number of iterations to execute when optimizing the size of exclusion radius.
        p : float, optional
            This is `p` argument of scipy.spatial.KDTree.query_ball_point method denoting
            which Minkowski p-norm to use. Should be in the range [1, inf]. A finite large p may
            cause a ValueError if overflow can occur.
        eps : nonnegative float, optional
            This is `eps` argument of scipy.spatial.KDTree.query_ball_point method denoting
            approximate nearest neighbor search for eliminating close points. Branches of the tree
            are not explored if their nearest points are further than r / (1 + eps), and branches
            are added in bulk if their furthest points are nearer than r * (1 + eps).
        random_seed : int, optional
            Seed for random selection of points be evaluated.
        ref_index: int, list, optional
            Index of the reference sample to start the selection algorithm from.
            It can be an integer, or a list of integers.
            When the `ref_index` is a list for multiple classes, it will be shared among all
            clusters. If we want to use different reference indices for each class, we can perform
            the subset selection for each class separately where different `ref_index` parameters
            can be used. For example, if we have two classes, we can pass `ref_index=[0, 1]` to
            select samples from class 0 and `ref_index=[3, 6]` class 1 respectively.
            Default is [0].
        fun_dist : callable, optional
            Function for calculating the pairwise distance between sample points to be used in
            calculating the medoid. `fun_dist(x) -> x_dist` takes a 2D feature array of shape
            (n_samples, n_features) and returns a 2D distance array of shape (n_samples, n_samples).

        """
        self.r0 = r0
        self.r = r0
        self.ref_index = ref_index
        self.n_iter = n_iter
        self.k = k
        self.tol = tol
        self.eps = eps
        self.p = p
        self.random_seed = random_seed
        self.fun_dist = fun_dist

    def algorithm(self, x, max_size) -> Union[List, Iterable]:
        """Return selected sample indices based on OptiSim algorithm.

        Parameters
        ----------
        x: ndarray of shape (n_samples, n_features)
            Feature matrix of `n_samples` samples in `n_features` dimensional feature space.
        max_size : int
            Maximum number of samples to select.

        Returns
        -------
        selected : Union[List, Iterable]
            List of indices of selected sample indices.

        """
        # set up reference index
        selected = get_initial_selection(x=x, x_dist=None, ref_index=self.ref_index, fun_dist=None)
        count = len(selected)

        # establish a kd-tree for nearest-neighbor lookup
        tree = spatial.KDTree(x)
        # use a random number generator that will be used to randomly select points
        rng = np.random.default_rng(seed=self.random_seed)

        n_samples = len(x)
        # bv will serve as a mask to discard points within radius r of previously selected points
        bv = np.zeros(n_samples)
        candidates = list(range(n_samples))
        # determine which points are within radius r of initial point
        # note: workers=-1 uses all available processors/CPUs
        index_remove = tree.query_ball_point(
            x[self.ref_index], self.r, eps=self.eps, p=self.p, workers=-1
        )
        # exclude points within radius r of initial point from list of candidates using bv mask
        for idx in index_remove:
            bv[idx] = 1
        candidates = np.ma.array(candidates, mask=bv)

        # while there are still remaining candidates to be selected
        # compressed returns all the non-masked data as a 1-D array
        while len(candidates.compressed()) > 0:
            # randomly select samples from list of candidates
            try:
                sublist = rng.choice(candidates.compressed(), size=self.k, replace=False)
            except ValueError:
                sublist = candidates.compressed()

            # create a new kd-tree for nearest neighbor lookup with candidates
            new_tree = spatial.KDTree(x[selected])
            # query the kd-tree for nearest neighbors to selected samples
            # note: workers=-1 uses all available processors/CPUs
            search, _ = new_tree.query(x[sublist], eps=self.eps, p=self.p, workers=-1)
            # identify the nearest neighbor with the largest distance from previously selected samples
            best_idx = sublist[np.argmax(search)]
            selected.append(best_idx)

            count += 1
            if count > max_size:
                # do this if you have reached the maximum number of points selected
                return selected

            # eliminate all samples within radius r of the selected sample
            index_remove = tree.query_ball_point(
                x[best_idx], self.r, eps=self.eps, p=self.p, workers=-1
            )
            for idx in index_remove:
                bv[idx] = 1
            candidates = np.ma.array(candidates, mask=bv)

        return selected

    def select_from_cluster(self, x, size, labels=None) -> Union[List, Iterable]:
        """Return selected samples from a cluster based on OptiSim algorithm.

        Parameters
        ----------
        x: ndarray of shape (n_samples, n_features)
            Feature matrix of `n_samples` samples in `n_features` dimensional feature space.
        size : int
            Number of samples to be selected.
        labels: np.ndarray
            Indices of samples that form a cluster.

        Returns
        -------
        selected : Union[List, Iterable]
            List of indices of selected samples.

        """
        if self.ref_index is not None and self.ref_index >= len(x):
            raise ValueError(
                f"ref_index is not less than the number of samples; {self.ref_index} >= {len(x)}."
            )
        # pass subset of x to optimize_radius if labels is not None
        if labels is not None:
            x = x[labels]
        # reset radius to initial value (this is important when sampling multiple clusters)
        self.r = self.r0
        return optimize_radius(self, x, size, labels)


class DISE(SelectionBase):
    """
    Select samples using Directed Sphere Exclusion (DISE) algorithm.

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

    References
    ----------
    Gobbi, A., and Lee, M.-L. (2002). DISE: directed sphere exclusion.
    Journal of Chemical Information and Computer Sciences,
    43(1), 317–323. https://doi.org/10.1021/ci025554v

    """

    def __init__(self, r0=None, ref_index=None, tol=0.05, n_iter=10, p=2.0, eps=0.0, fun_dist=None):
        """
        Initialize class.

        Parameters
        ----------
        r0: float, optional
            Initial guess for radius of the exclusion sphere.
        ref_index: int, list, optional
            Index of the reference sample to start the selection algorithm from.
            It can be an integer, or a list of integers or None. When None, the medoid center is
            chosen as the reference sample.
            When the `ref_index` is a list for multiple classes,
            it will be shared among all clusters.
            If we want to use different reference indices for each class, we can perform the subset
            selection for each class separately where different `ref_index` parameters can be used.
            For example, if we have two classes, we can pass `ref_index=[0, 1]` to select samples
            from class 0 and `ref_index=[3, 6]` class 1 respectively.
        tol: float, optional
            Percentage tolerance of sample size error. Given a subset size, the selected size
            will be within size * (1 - tol) and size * (1 + tol).
        n_iter: int, optional
            Number of iterations for optimizing the radius of exclusion sphere.
        p: float, optional
            This is `p` argument of scipy.spatial.KDTree.query_ball_point method denoting
            which Minkowski p-norm to use. The values of `p` should be within [1, inf].
            A finite large p may cause a ValueError if overflow can occur. Default is 2.0.
        eps: nonnegative float, optional
            This is `eps` argument of scipy.spatial.KDTree.query_ball_point method denoting
            approximate nearest neighbor search for eliminating close points. Branches of the tree
            are not explored if their nearest points are further than r / (1 + eps), and branches
            are added in bulk if their furthest points are nearer than r * (1 + eps).
        fun_dist: callable, optional
            Function for calculating the distances between sample points. When `fun_dist` is `None`,
            the Minkowski p-norm distance is used. Default is None.

        """
        self.r0 = r0
        self.r = r0
        self.ref_index = ref_index
        self.tol = tol
        self.n_iter = n_iter
        self.p = p
        self.eps = eps

        # if fun_dist is None:
        #     self.fun_dist = spatial.distance.pdist
        # else:
        #     self.fun_dist = fun_dist
        self.fun_dist = fun_dist

    def algorithm(self, x, max_size) -> Union[List, Iterable]:
        """Return selected samples based on directed sphere exclusion algorithm.

        Parameters
        ----------
        x: ndarray of shape (n_samples, n_features)
           Feature matrix of `n_samples` samples in `n_features` dimensional space.
        max_size: int
            Maximum number of samples to select.

        Returns
        -------
        selected: Union[List, Iterable]
            List of indices of selected samples.

        """
        if self.fun_dist is None:
            distances = spatial.distance.squareform(
                spatial.distance.pdist(x, metric="minkowski", p=self.p)
            )
        else:
            distances = self.fun_dist(x)

        # set up the ref_index as when is None
        if self.ref_index is None:
            self.ref_index = get_initial_selection(
                x=None,
                x_dist=distances,
                ref_index=self.ref_index,
                fun_dist=None,
            )
        # set up the ref_index for integer and list of integers
        elif isinstance(self.ref_index, (int, list)):
            self.ref_index = get_initial_selection(
                x=x,
                x_dist=distances,
                ref_index=self.ref_index,
                fun_dist=None,
            )
        # not supported ref_index
        else:
            raise ValueError(
                "The provided reference indices are not supported in the current implementation."
            )

        # # calculate distance of all samples from reference sample; distance is a (n_samples,) array
        # # this includes the distance of reference sample from itself, which is 0
        # distances = spatial.minkowski_distance(x[self.ref_index], x, p=self.p)
        distances_ref = distances[self.ref_index[0], :]

        # get sorted index of samples based on their distance from reference (closest to farthest)
        # the first index will be the ref_index which has distance of zero
        index_sorted = np.argsort(distances_ref)
        assert index_sorted[0] == self.ref_index
        # construct KDTree for quick nearest-neighbor lookup
        kdtree = spatial.KDTree(x)

        # construct bitarray to track selected samples (1 means exclude)
        bv = bitarray.bitarray(list(np.zeros(len(x), dtype=int)))

        # the neighbours of the ref_index are going to be excluded in the first iteration
        # and ref_index is going to be added to the selected list
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
                    x[idx], self.r, eps=self.eps, p=self.p, workers=-1
                )
                # exclude samples within radius r of sample idx (measure by Minkowski p-norm) from
                # future consideration by setting their bitarray value to 1
                for index in index_exclude:
                    bv[index] = 1

        return selected

    def select_from_cluster(self, x, size, labels=None) -> Union[List, Iterable]:
        """Return selected samples from a cluster based on directed sphere exclusion algorithm

        Parameters
        ----------
        x: ndarray of shape (n_samples, n_features)
           Feature matrix of `n_samples` samples in `n_features` dimensional space.
        size: int
            Number of samples to be selected.
        labels: np.ndarray, optional
            Indices of samples that form a cluster.

        Returns
        -------
        selected: Union[List, Iterable]
            List of indices of selected samples.

        """
        # pass subset of x to optimize_radius if labels is not None
        if labels is not None:
            x = x[labels]

        if x.shape[0] < size:
            raise RuntimeError(
                f"Number of samples is less than the requested "
                f"sample size: {x.shape[0]} < {size}."
            )
        # reset radius to initial value (this is important when sampling multiple clusters)
        self.r = self.r0
        return optimize_radius(self, x, size, labels)


def get_initial_selection(x=None, x_dist=None, ref_index=None, fun_dist=None) -> List:
    """Set up the reference index for selecting.

    Parameters
    ----------
    x: ndarray of shape (n_samples, n_features), optional
        Feature matrix of `n_samples` samples in `n_features` dimensional feature space.
    x_dist: ndarray of shape (n_samples, n_samples), optional
        Pairwise distance matrix between `n_samples` samples.
    ref_index: int, list, optional
        Index of the reference sample to start the selection algorithm from.
        It can be an integer, or a list of integers or None. When None, the medoid center is chosen as the reference
        sample.
        When the `ref_index` is a list for multiple classes, it will be shared among all clusters.
        If we want to use different reference indices for each class, we can perform the subset
        selection for each class separately where different `ref_index` parameters can be used.
        For example, if we have two classes, we can pass `ref_index=[0, 1]` to select samples
        from class 0 and `ref_index=[3, 6]` class 1 respectively.
    func_dist: callable, optional
        Function for calculating the pairwise distance between sample points to be used in
        calculating the medoid. `func_dist(x) -> x_dist` takes a 2D feature array of shape
        (n_samples, n_features) and returns a 2D distance array of shape (n_samples, n_samples).

    Returns
    -------
    initial_selections: List
        List of indices of the initial selected data points.

    """
    # use the medoid center as the reference sample if ref_index is None
    if ref_index is None:
        if x_dist is None:
            x_dist = fun_dist(x)
        # calculate the medoid center
        initial_selections = [int(np.argmin(np.sum(x_dist, axis=0)))]

    # the length of the distance matrix is the number of samples
    if x_dist is not None:
        len_x = len(x_dist)
    else:
        len_x = len(x)

    # when ref_index is an integer, it cannot be negative or greater than the number of samples
    if isinstance(ref_index, int):
        # check if ref_index is a valid index
        if ref_index < 0 or ref_index >= len_x:
            raise ValueError(
                f"The ref_index must be a non-negative integer less than the "
                f"number of samples, got {ref_index} >= {len_x}."
            )
        initial_selections = [int(ref_index)]

    # when ref_index is a list, just use it
    if isinstance(ref_index, list):
        # all elements of ref_index must be integers or float
        if not all(isinstance(i, (int)) for i in ref_index):
            raise ValueError("All elements of ref_index must be integers.")

        # all the elements of ref_index must be greater than or equal to 0 and less than the
        # number of samples
        if np.any(np.array(ref_index) < 0) or np.any(np.array(ref_index) >= len_x):
            raise ValueError(
                f"All elements of ref_index must be greater than or equal to 0 and less than "
                f"the number of samples, got {ref_index}."
            )

        initial_selections = [int(i) for i in ref_index]

    return initial_selections
