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

"""Module for Selection Utilities."""

import numpy as np


__all__ = [
    "predict_radius",
]


def predict_radius(obj, arr, num_selected, cluster_ids=None):
    """Algorithm that uses sphere_exclusion for selecting points from cluster.

    Parameters
    ----------
    obj: object
        Instance of dissimilarity selection class
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
    # set the limits on # of selected points according to the tolerance percentage
    error = num_selected * obj.tolerance / 100
    lowlimit = num_selected - error
    uplimit = num_selected + error
    # sort into clusters if data is labelled
    if cluster_ids is not None:
        arr = arr[cluster_ids]

    original_r = None
    # use specified radius if passed in
    if obj.r is not None:
        original_r = obj.r
        # run the selection algorithm
        result = obj.algorithm(arr, uplimit)
    # calculate radius from largest feature values
    else:
        rg = max(np.ptp(arr, axis=0)) / num_selected * 3
        obj.r = rg
        result = obj.algorithm(arr, uplimit)

    # correct number of points chosen, return selected
    if len(result) == num_selected:
        return result

    # incorrect number of points chosen
    low = obj.r if len(result) > num_selected else 0
    high = obj.r if low == 0 else None
    bounds = [low, high]
    count = 0
    while (len(result) < lowlimit or len(result) > uplimit) and count < 10:
        # too many points selected, make radius larger
        if bounds[1] is None:
            rg = bounds[0] * 2
        # too few points selected, make radius smaller
        else:
            rg = (bounds[0] + bounds[1]) / 2
        obj.r = rg
        # re-run selection with new radius
        result = obj.algorithm(arr, uplimit)

        # adjust upper/lower bounds of radius size to fine tune
        if len(result) > num_selected:
            bounds[0] = rg
        else:
            bounds[1] = rg
        count += 1
    # cannot find radius that produces desired number of selected points
    if count >= 10:
        print(f"Optimal radius finder failed to converge, selected {len(result)} molecules instead "
              f"of requested {num_selected}.")
    # undo any changes to radius
    obj.r = original_r
    return result
