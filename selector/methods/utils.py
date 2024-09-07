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
"""Module for Selection Utilities."""

import warnings

import numpy as np

__all__ = [
    "optimize_radius",
]


def optimize_radius(obj, X, size, cluster_ids=None):
    """Algorithm that uses sphere exclusion for selecting points from cluster.

    Iteratively searches for the optimal radius to obtain the correct number
    of selected samples. If the radius cannot converge to return `size` points,
    the function returns the closest number of samples to `size` as possible.

    Parameters
    ----------
    obj: object
        Instance of `DirectedSphereExclusion` or `OptiSim` selection class.
    X: ndarray of shape (n_samples, n_features)
        Feature matrix of `n_samples` samples in `n_features` dimensional space.
    size: int
        Number of sample points to select (i.e. size of the subset).
    cluster_ids: np.ndarray
        Indices of points that form a cluster.

    Returns
    -------
    selected: list
        List of indices of selected samples.
    """
    if X.shape[0] < size:
        raise RuntimeError(
            f"Size of samples to be selected is greater than existing the number of samples; "
            f"{size} > {X.shape[0]}."
        )
    # set the limits on # of selected points according to the tolerance percentage
    error = size * obj.tol
    lower_size = round(size - error)
    upper_size = round(size + error)

    # select `upper_size` number of samples
    if obj.r is not None:
        # use initial sphere radius
        selected = obj.algorithm(X, upper_size)
    else:
        # calculate a sphere radius based on maximum of n_features range
        # np.ptp returns range of values (maximum - minimum) along an axis
        obj.r = max(np.ptp(X, axis=0)) / size * 3
        selected = obj.algorithm(X, upper_size)

    # return selected if the correct number of samples chosen
    if len(selected) == size:
        return selected

    # optimize radius to select the correct number of samples
    # first, set a sensible range for optimizing r value within that range
    if len(selected) > size:
        # radius should become bigger, b/c too many samples were selected
        bounds = [obj.r, np.inf]
    else:
        # radius should become smaller, b/c too few samples were selected
        bounds = [0, obj.r]

    n_iter = 0
    while (len(selected) < lower_size or len(selected) > upper_size) and n_iter < obj.n_iter:
        # change sphere radius based on the defined bound
        if bounds[1] == np.inf:
            # make sphere radius larger by a factor of 2
            obj.r = bounds[0] * 2
        else:
            # make sphere radius smaller by a factor of 1/2
            obj.r = (bounds[0] + bounds[1]) / 2

        # re-select samples with the new radius
        selected = obj.algorithm(X, upper_size)

        # adjust lower/upper bounds of radius range
        if len(selected) > size:
            bounds[0] = obj.r
        else:
            bounds[1] = obj.r
        n_iter += 1

    # cannot find radius that produces desired number of selected points
    if n_iter >= obj.n_iter and len(selected) != size:
        warnings.warn(
            f"Optimal radius finder failed to converge, selected {len(selected)} points instead "
            f"of requested {size}."
        )

    return selected
