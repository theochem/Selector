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
from DiverseSelector.methods.utils import predict_radius
import numpy as np
from scipy import spatial


__all__ = [
    "MaxMin",
    "MaxSum",
    "OptiSim",
]


class MaxMin(SelectionBase):
    """Selecting compounds using MaxMin algorithm.

    Initial point is chosen as medoid center. The second point is
    the furthest point away. All the following points are selected
    using the rule:
    1. Find minimum distance from every point to the selected ones.
    2. Select a point the has the maximum distance among calculated
       on the previous step.

    The algorithm is described well here: https://doi.org/10.1002/qsar.200290002
    """

    def __init__(self, func_distance=None):
        """
        Initializing class.

        Parameters
        ----------
        func_distance: callable
            Function for calculating the pairwise distance between instances of the array.
        """
        self.func_distance = func_distance

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm MaxMin for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Distance matrix for points that needs to be selected if func_distance is None.
            Otherwise, treated as coordinates array.
        num_selected: int
            Number of molecules that need to be selected
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if self.func_distance is not None:
            # calculate pairwise distance between instances of provided array
            arr_dist = self.func_distance(arr)
        else:
            # assume provided array already describes pairwise distances between samples
            arr_dist = arr

        if cluster_ids is not None:
            # extract pairwise distances from full pairwise distance matrix to obtain a new matrix that only contains
            # pairwise distances between samples within a given cluster
            arr_dist = arr_dist[cluster_ids][:, cluster_ids]

        # choosing initial point as the medoid (i.e., point with minimum cumulative pairwise distances to other points)
        selected = [np.argmin(np.sum(arr_dist, axis=0))]
        # select following points until desired number of points have been obtained
        while len(selected) < num_selected:
            # determine the minimum pairwise distances between the selected points and all the other points
            min_distances = np.min(arr_dist[selected], axis=0)
            # determine which point affords the maximum distance among the minimum distances captured in min_distances
            new_id = np.argmax(min_distances)
            selected.append(new_id)
        return selected


class MaxSum(SelectionBase):
    """Selecting compounds using MaxSum algorithm.

    Initial point is chosen as medoid center. The second point is
    the furthest point away. All the following points are selected
    using the rule:
    1. Determine the sum of distances from every point to the selected ones.
    2. Select a point the has the maximum sum of distance among calculated
       on the previous step.
    """

    def __init__(self, func_distance=None):
        """
        Initializing class.

        Parameters
        ----------
        func_distance: callable
            Function for calculating the pairwise distance between instances of the array.
        """
        self.func_distance = func_distance

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm MaxSum for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Distance matrix for points that needs to be selected if func_distance is None.
            Otherwise, treated as coordinates array.
        num_selected: int
            Number of molecules that need to be selected
        cluster_ids: np.ndarray
            Indices of molecules that form a cluster

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        if num_selected > len(arr):
            raise ValueError(f"Requested {num_selected} points which is greater than {len(arr)} "
                             f"points provided in array")

        if self.func_distance is not None:
            # calculate pairwise distance between instances of provided array
            arr_dist = self.func_distance(arr)
        else:
            # assume provided array already describes pairwise distances between samples
            arr_dist = arr

        if cluster_ids is not None:
            # extract pairwise distances from full pairwise distance matrix to obtain a new matrix that only contains
            # pairwise distances between samples within a given cluster
            arr_dist = arr_dist[cluster_ids][:, cluster_ids]

        # choosing initial point as the medoid (i.e., point with minimum cumulative pairwise distances to other points)
        selected = [np.argmin(np.sum(arr_dist, axis=0))]
        # select following points until desired number of points have been obtained
        while len(selected) < num_selected:
            # determine sum of pairwise distances between selected points and all other points
            sum_distances = np.sum(arr_dist[selected], axis=0)
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
    """Selecting compounds using OptiSim algorithm.

    Initial point is chosen as medoid center. Points are randomly chosen and added to a subsample
    if outside of radius r from all previously selected points, and discarded otherwise. Once k
    number of points are added to the subsample, the point with the greatest minimum distance to
    previously selected points is selected and the subsample is cleared and the process repeats.

    Adapted from  https://doi.org/10.1021/ci970282v
    """

    def __init__(self, r=None, k=10, tolerance=5.0, eps=0, p=2, start_id=0, random_seed=42):
        """
        Initializing class.

        Parameters
        ----------
        r: float
            Initial guess of radius for optisim algorithm, no points within r distance to an already
            selected point can be selected.
        k: int
            Amount of points to add to subsample before selecting one of the points with the
            greatest minimum distance to the previously selected points.
        tolerance: float
            Percentage error of number of molecules actually selected from number of molecules
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
        self.k = k
        self.tolerance = tolerance
        self.eps = eps
        self.p = p
        self.start_id = start_id
        self.random_seed = random_seed

    def algorithm(self, arr, uplimit) -> list:
        """
        Optisim algorithm logic.

        Parameters
        ----------
        arr: np.ndarray
            Coordinate array of points.
        uplimit: int
            Maximum number of points to select.

        Returns
        -------
        selected: list
            List of ids of selected molecules
        """
        selected = [self.start_id]
        count = 1

        # establish a kd-tree for nearest-neighbor lookup
        tree = spatial.KDTree(arr)
        # use a random number generator that will be used to randomly select points
        rng = np.random.default_rng(seed=self.random_seed)

        len_arr = len(arr)
        # bv will serve as a mask to discard points within radius r of previously selected points
        bv = np.zeros(len_arr)
        candidates = list(range(len_arr))
        # determine which points are within radius r of initial point
        elim = tree.query_ball_point(arr[self.start_id], self.r, eps=self.eps, p=self.p, workers=-1)
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
            newtree = spatial.KDTree(arr[selected])
            # query the kd-tree for nearest neighbors to selected samples
            search, _ = newtree.query(arr[sublist], eps=self.eps, p=self.p, workers=-1)
            # identify the nearest neighbor with the largest distance from previously selected samples
            search_idx = np.argmax(search)
            best_idx = sublist[search_idx]
            selected.append(best_idx)

            count += 1
            if count > uplimit:
                # do this if you have reached the maximum number of points selected
                return selected

            # eliminate all remaining candidates within radius r of the sample that was just selected
            elim = tree.query_ball_point(arr[best_idx], self.r, eps=self.eps, p=self.p, workers=-1)
            for idx in elim:
                bv[idx] = 1
            candidates = np.ma.array(candidates, mask=bv)

        return selected

    def select_from_cluster(self, arr, num_selected, cluster_ids=None):
        """
        Algorithm that uses optisim for selecting points from cluster.

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
        return predict_radius(self, arr, num_selected, cluster_ids)


