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
import warnings

import numpy as np


__all__ = [
    "optimize_radius",
]


def optimize_radius(obj, x, num_selected, cluster_ids=None):
    """Algorithm that uses sphere exclusion for selecting points from cluster.

    Iteratively searches for the optimal radius to obtain the correct number
    of selected points. If the radius cannot converge to return `num_selected` points,
    the function returns the closest number of points to `num_selected` as possible.

    Parameters
    ----------
    obj: object
        Instance of 'DirectedSphereExclusion' or 'OptiSim' selection class.
    x: np.ndarray
        Feature array.
    num_selected: int
        Number of points that need to be selected.
    cluster_ids: np.ndarray
        Indices of points that form a cluster.

    Returns
    -------
    selected: list
        List of ids of selected points.
    """
    if x.shape[0] < num_selected:
        raise RuntimeError(
            f"The number of selected points {num_selected} is greater than the number of points"
            f"provided {x.shape[0]}."
        )
    # set the limits on # of selected points according to the tolerance percentage
    error = num_selected * obj.tolerance
    lowlimit = round(num_selected - error)
    uplimit = round(num_selected + error)
    # sort into clusters if data is labelled
    if cluster_ids is not None:
        x = x[cluster_ids]

    # use specified radius if passed in
    if obj.r is not None:
        # run the selection algorithm
        result = obj.algorithm(x, uplimit)
    # calculate radius from largest feature values
    else:
        rg = max(np.ptp(x, axis=0)) / num_selected * 3
        obj.r = rg
        result = obj.algorithm(x, uplimit)

    # correct number of points chosen, return selected
    if len(result) == num_selected:
        return result

    # if incorrect number of points chosen, then optimize radius
    if len(result) > num_selected:
        # Too many points, so the radius should be bigger.
        bounds = [obj.r, np.inf]
    else:
        bounds = [0, obj.r]
    niter = 0
    print(f"Number of results {len(result)}, {lowlimit}, {uplimit}")
    while (len(result) < lowlimit or len(result) > uplimit) and niter < 10:
        # too many points selected, make radius larger
        if bounds[1] == np.inf:
            rg = bounds[0] * 2
        # too few points selected, make radius smaller
        else:
            rg = (bounds[0] + bounds[1]) / 2
        obj.r = rg
        # re-run selection with new radius
        result = obj.algorithm(x, uplimit)

        # adjust upper/lower bounds of radius size to fine tune
        if len(result) > num_selected:
            bounds[0] = rg
        else:
            bounds[1] = rg
        print(f"Radius ", rg)
        niter += 1
    # cannot find radius that produces desired number of selected points
    if niter >= 10:
        warnings.warn(
            f"Optimal radius finder failed to converge, selected {len(result)} molecules instead "
              f"of requested {num_selected}."
        )

    print(f"radius after optimization: ", obj.r)
    return result
