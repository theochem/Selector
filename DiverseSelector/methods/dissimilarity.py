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

"""Module for Dissimilarity-Based Selection Methods."""

from DiverseSelector.methods.base import SelectionBase
from DiverseSelector.methods.utils import optimize_radius
import numpy as np
from scipy import spatial


__all__ = [
    "MaxMin",
    "MaxSum",
    "OptiSim",
]


class MaxMin(SelectionBase):
    """Selecting samples using MaxMin algorithm.

    MaxMin is possibly the most widely used method for dissimilarity-based
    compound selection. When presented with a dataset of samples, the
    initial point is chosen as the dataset's medoid center. Next, the second
    point is chosen to be that which is furthest from this initial point.
    Subsequently, all following points are selected via the following
    logic:

    1. Find the minimum distance from every point to the already-selected ones.
    2. Select the point which has the maximum distance among those calculated
       in the previous step.

    References
    ----------
    [1] Ashton, Mark, et al., Identification of diverse database subsets using 
    property‐based and fragment‐based molecular descriptions, Quantitative 
    Structure‐Activity Relationships 21.6 (2002): 598-604.
    """

    def __init__(self, func_distance=None):
        """
        Initializing class.

        Parameters
        ----------
        func_distance : callable
            Function for calculating the pairwise distance between instances of the coordinates array.
        """
        self.func_distance = func_distance

    def select_from_cluster(self, X, size, cluster_ids=None):
        """Return selected samples from a cluster based on MaxMin algorithm.

        Parameters
        ----------
        X : np.ndarray
            Distance matrix for points that needs to be selected if func_distance is None.
            Otherwise, treated as coordinates array.
        size : int
            Number of samples to be selected.
        cluster_ids : np.ndarray
            Indices of samples that form a cluster.

        Returns
        -------
        selected : list
            List of indices of selected samples.
        """
        if self.func_distance is not None:
            # calculate pairwise distance between instances of provided array
            X_dist = self.func_distance(X)
        else:
            # assume provided array already describes pairwise distances between samples
            X_dist = X

        if cluster_ids is not None:
            # extract pairwise distances from full pairwise distance matrix to obtain a new matrix that only contains
            # pairwise distances between samples within a given cluster
            X_dist = X_dist[cluster_ids][:, cluster_ids]

        # choosing initial point as the medoid (i.e., point with minimum cumulative pairwise distances to other points)
        selected = [np.argmin(np.sum(X_dist, axis=0))]
        # select following points until desired number of points have been obtained
        while len(selected) < size:
            # determine the minimum pairwise distances between the selected points and all the other points
            min_distances = np.min(X_dist[selected], axis=0)
            # determine which point affords the maximum distance among the minimum distances captured in min_distances
            new_id = np.argmax(min_distances)
            selected.append(new_id)
        return selected


class MaxSum(SelectionBase):
    """Selecting samples using MaxSum algorithm.

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

    def __init__(self, func_distance=None):
        """
        Initializing class.

        Parameters
        ----------
        func_distance : callable
            Function for calculating the pairwise distance between instances of the coordinates array.
        """
        self.func_distance = func_distance

    def select_from_cluster(self, X, size, cluster_ids=None):
        """Return selected samples from a cluster based on MaxSum algorithm.

        Parameters
        ----------
        X : np.ndarray
            Distance matrix for points that needs to be selected if func_distance is None.
            Otherwise, treated as coordinates array.
        size : int
            Number of samples to be selected.
        cluster_ids : np.ndarray
            Indices of samples that form a cluster.

        Returns
        -------
        selected : list
            List of indices of selected samples.
        """
        if size > len(X):
            raise ValueError(f"Requested {size} points which is greater than {len(X)} "
                             f"points provided in array")

        if self.func_distance is not None:
            # calculate pairwise distance between instances of provided array
            X_dist = self.func_distance(X)
        else:
            # assume provided array already describes pairwise distances between samples
            X_dist = X

        if cluster_ids is not None:
            # extract pairwise distances from full pairwise distance matrix to obtain a new matrix that only contains
            # pairwise distances between samples within a given cluster
            X_dist = X_dist[cluster_ids][:, cluster_ids]

        # choosing initial point as the medoid (i.e., point with minimum cumulative pairwise distances to other points)
        selected = [np.argmin(np.sum(X_dist, axis=0))]
        # select following points until desired number of points have been obtained
        while len(selected) < size:
            # determine sum of pairwise distances between selected points and all other points
            sum_distances = np.sum(X_dist[selected], axis=0)
            while True:
                # determine which new point has the maximum sum of pairwise distances with already-selected points
                new_id = np.argmax(sum_distances)
                if new_id in selected:
                    sum_distances[new_id] = 0
                else:
                    break
            selected.append(new_id)
        return selected


class OptiSim(SelectionBase):
    """Selecting samples using OptiSim algorithm.

    The OptiSim algorithm selects samples from a dataset by first choosing the medoid center as the
    initial point. Next, points are randomly chosen and added to a subsample if they exist
    outside of radius r from all previously selected points (otherwise, they are discarded). Once k
    number of points have been added to the subsample, the point with the greatest minimum distance to
    the previously selected points is chosen. Then, the subsample is cleared and the process is repeated.

    References
    ----------
    [1] J. Chem. Inf. Comput. Sci. 1997, 37, 6, 1181–1188. https://doi.org/10.1021/ci970282v
    """

    def __init__(self, r0=None, k=10, tol=5.0, eps=0, p=2, start_id=0, random_seed=42, n_iter=10):
        """Initialize class.

        Parameters
        ----------
        r0 : float
            Initial guess of radius for OptiSim algorithm. No points within r distance to an already
            selected point can be selected.
        k : int
            Amount of points to add to subsample before selecting one of the points with the
            greatest minimum distance to the previously selected points.
        tol : float
            Percentage error of number of samples actually selected from number of samples
            requested.
        eps : float
            Approximate nearest neighbor search for eliminating close points. Branches of the tree
            are not explored if their nearest points are further than r / (1 + eps), and branches
            are added in bulk if their furthest points are nearer than r * (1 + eps).
        p : float
            Which Minkowski p-norm to use. Should be in the range [1, inf]. A finite large p may
            cause a ValueError if overflow can occur.
        start_id : int
            Index for the first point to be selected.
        random_seed : int
            Seed for random selection of points be evaluated.
        n_iter : int
            Number of iterations to execute when optimizing the size of exclusion radius. Default is 10.
        """
        self.r = r0
        self.k = k
        self.tol = tol
        self.eps = eps
        self.p = p
        self.start_id = start_id
        self.random_seed = random_seed
        self.n_iter = n_iter

    def algorithm(self, X, max_size) -> list:
        """Return selected samples based on OptiSim algorithm.

        Parameters
        ----------
        X : np.ndarray
            Coordinate array of samples.
        max_size : int
            Maximum number of samples to select.

        Returns
        -------
        selected : list
            List of indices of selected samples.
        """
        selected = [self.start_id]
        count = 1

        # establish a kd-tree for nearest-neighbor lookup
        tree = spatial.KDTree(X)
        # use a random number generator that will be used to randomly select points
        rng = np.random.default_rng(seed=self.random_seed)

        len_X = len(X)
        # bv will serve as a mask to discard points within radius r of previously selected points
        bv = np.zeros(len_X)
        candidates = list(range(len_X))
        # determine which points are within radius r of initial point
        elim = tree.query_ball_point(X[self.start_id], self.r, eps=self.eps, p=self.p, workers=-1)
        # exclude points within radius r of initial point from list of candidates using bv mask
        for idx in elim:
            bv[idx] = 1
        candidates = np.ma.array(candidates, mask=bv)

        # while there are still remaining candidates to be selected
        while len(candidates.compressed()) > 0:
            # randomly select samples from list of candidates
            try:
                sublist = rng.choice(candidates.compressed(), size=self.k, replace=False)
            except ValueError:
                sublist = candidates.compressed()

            # create a new kd-tree for nearest neighbor lookup with candidates
            newtree = spatial.KDTree(X[selected])
            # query the kd-tree for nearest neighbors to selected samples
            search, _ = newtree.query(X[sublist], eps=self.eps, p=self.p, workers=-1)
            # identify the nearest neighbor with the largest distance from previously selected samples
            search_idx = np.argmax(search)
            best_idx = sublist[search_idx]
            selected.append(best_idx)

            count += 1
            if count > max_size:
                # do this if you have reached the maximum number of points selected
                return selected

            # eliminate all remaining candidates within radius r of the sample that was just selected
            elim = tree.query_ball_point(X[best_idx], self.r, eps=self.eps, p=self.p, workers=-1)
            for idx in elim:
                bv[idx] = 1
            candidates = np.ma.array(candidates, mask=bv)

        return selected

    def select_from_cluster(self, X, size, cluster_ids=None):
        """Return selected samples from a cluster based on OptiSim algorithm.

        Parameters
        ----------
        X : np.ndarray
            Coordinate array of samples.
        size : int
            Number of samples to be selected.
        cluster_ids : np.ndarray
            Indices of samples that form a cluster.

        Returns
        -------
        selected : list
            List of indices of selected samples.
        """
        return optimize_radius(self, X, size, cluster_ids)


