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

"""Utils module."""

import numpy as np


__all__ = [
    "sim_to_dist",
    "distance_to_similarity",
    "reverse",
    "reciprocal",
    "exponential",
    "gaussian",

]


def sim_to_dist(x: np.ndarray, metric) -> np.ndarray:
    """Convert similarity coefficients to distance matrix.

    Parameters
    ----------
    x : ndarray
        Symmetric similarity array.
    metric : str, int
        String or integer specifying which conversion metric to use.
        Supported metrics are "reverse", "reciprocal", "exp" (exponential),
        "gaussian", and "membership". If the metric is an integer,
        integer value distance conversion is performed.

    Returns
    -------
    y : ndarray
        Symmetric distance or similarity array.

    """
    method_dict = {
        "reverse": reverse,
        "reciprocal": reciprocal,
        "exp": exponential,
        "gaussian": gaussian,
    }
    if type(metric) == str:
        if metric in method_dict:
            return method_dict[metric](x)
        elif metric == "membership":
            return 1 - x
    elif type(metric) == int:
        return metric - x
    else:
        raise ValueError(f"{metric} is an unsupported metric.")

    return np.eye(5)


def reverse(x: np.ndarray):
    r"""Calculate distance matrix from similarity using the 'reverse' method.

    .. math::
    \delta_{ij} = min(s_{ij}) + max(s_{ij}) - s_{ij}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, :math:`s_{ij}` is their similarity coefficient,
    and :math:`max` and :math:`min` are the maximum and minimum
    values across the entire similarity matrix.

    Parameters
    -----------
    x : ndarray
        Symmetric similarity array.

    Returns
    -------
    ndarray :
        Symmetric distance array.

    """
    rev = (np.max(x) + np.min(x)) - x
    return rev


def reciprocal(x: np.ndarray):
    r"""Calculate distance matrix from similarity using the 'reverse' method.

    .. math::
    \delta_{ij} = \frac{1}{s_{ij}}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, and :math:`s_{ij}` is their similarity coefficient.

    Parameters
    -----------
    x : ndarray
        Symmetric similarity array.

    Returns
    -------
    ndarray :
        Symmetric distance array.
    """

    return 1/x


def exponential(x: np.ndarray):
    r"""Calculate distance matrix from similarity using the 'reverse' method.

    .. math::
    \delta_{ij} = -\ln{\frac{s_{ij}}{max(s_{ij})}}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, and :math:`s_{ij}` is their similarity coefficient.

    Parameters
    -----------
    x : ndarray
        Symmetric similarity array.

    Returns
    -------
    ndarray :
        Symmetric distance array.

    """
    print(f"max of x: ", np.max(x))
    y = x / (np.max(x))
    print(y)
    exp = -np.log(y)
    return exp


def gaussian(x: np.ndarray):
    """Calculate distance matrix from similarity using the 'reverse' method.

    Parameters
    -----------
    x : ndarray
        Symmetric similarity array.

    Returns
    -------
    ndarray :
        Symmetric distance array.

    """
    pass


def distance_to_similarity(x: np.ndarray, dist: bool = True) -> np.ndarray:
    """Convert between distance and similarity matrix.

    Parameters
    ----------
    x : ndarray
        Symmetric distance or similarity array.
    dist : bool
        Confirms the matrix is distance.

    Returns
    -------
    y : ndarray
        Symmetric distance or similarity array.
    """
    if dist is True:
        y = 1 / (1 + x)
    else:
        y = (1 / x) - 1
    return y
