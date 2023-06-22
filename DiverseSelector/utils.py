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
    "correlation",
    "transition",
    "co_occur",
    "gravity",
    "probability",
    "covariance",
]


def sim_to_dist(x: np.ndarray, metric):
    """Convert similarity coefficients to distance matrix.

    Parameters
    ----------
    x : ndarray
        Symmetric similarity array.
    metric : str, int
        String or integer specifying which conversion metric to use.
        Supported metrics are "reverse", "reciprocal", "exponential",
        "gaussian", "membership", "correlation", "transition", "co-occurrence",
        "gravity", "confusion", "probability", and "covariance". If the metric is an integer,
        the similarities are subtracted from that integer and the result is
        returned as distance.

    Returns
    -------
    ndarray :
        Symmetric distance array.
    """
    method_dict = {
        "reverse": reverse,
        "reciprocal": reciprocal,
        "exponential": exponential,
        "gaussian": gaussian,
        "correlation": correlation,
        "transition": transition,
        "co-occurrence": co_occur,
        "gravity": gravity,
        "probability": probability,
        "covariance": covariance,
    }

    if type(metric) == str:
        if metric in method_dict:
            return method_dict[metric](x)
        elif metric == "membership" or metric == "confusion":
            if np.any(x < 0) or np.any(x > 1):
                raise ValueError(
                    "There is an out of bounds value. Please make"
                    "sure all input values are between [0, 1]."
                )
            return 1 - x
    elif type(metric) == int:
        return metric - x
    else:
        raise ValueError(f"{metric} is an unsupported metric.")

    return


def reverse(x: np.ndarray):
    r"""Calculate distance matrix from similarity using the reverse method.

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
    dist = (np.max(x) + np.min(x)) - x
    return dist


def reciprocal(x: np.ndarray):
    r"""Calculate distance matrix from similarity using the reciprocal method.

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

    return 1 / x


def exponential(x: np.ndarray):
    r"""Calculate distance matrix from similarity using the exponential method.

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
    y = x / (np.max(x))
    dist = -np.log(y)
    return dist


def gaussian(x: np.ndarray):
    r"""Calculate distance matrix from similarity using the Gaussian method.

    .. math::
    \delta_{ij} = \sqrt{-\ln{\frac{s_{ij}}{max(s_{ij})}}}

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
    y = x / (np.max(x))
    dist = np.sqrt(-np.log(y))
    return dist


def correlation(x: np.ndarray):
    r"""Calculate distance matrix from correlation matrix.

    .. math::
    \delta_{ij} = \sqrt{1 - r_{ij}}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, and :math:`r_{ij}` is their correlation.

    Parameters
    -----------
    x : ndarray
        Symmetric correlation array.

    Returns
    -------
    ndarray :
        Symmetric distance array.

    """
    if np.any(x < -1) or np.any(x > 1):
        raise ValueError(
            "There is an out of bounds value. Please make"
            "sure all correlations are between [-1, 1]."
        )
    dist = np.sqrt(1 - x)
    return dist


def transition(x: np.ndarray):
    r"""Calculate distance matrix from frequency using the transition method.

    .. math::
    \delta_{ij} = \frac{1}{\sqrt{f_{ij}}}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, and :math:`f_{ij}` is their frequency.

    Parameters
    -----------
    x : ndarray
        Symmetric frequency array.

    Returns
    -------
    ndarray :
        Symmetric distance array.

    """
    if np.any(x < 0):
        raise ValueError(
            "There is a negative value in the input. Please"
            "make sure all frequency values are non-negative."
        )

    # ignore divide by zero warnings
    with np.errstate(divide="ignore"):
        dist = 1 / np.sqrt(x)
    # replace all divide by zero elements with value 0
    dist[dist == np.inf] = 0
    return dist


def co_occur(x: np.ndarray):
    r"""Calculate distance matrix from frequency using the co-occurrence method.

    .. math::
    \delta_{ij} =  \left(1 + \frac{f_{ij}\sum_{i,j}{f_{ij}}}
    {\sum_{i}{f_{ij}}\sum_{j}{f_{ij}}} \right)^{-1}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, and :math:`f_{ij}` is their frequency.

    Parameters
    -----------
    x : ndarray
        Symmetric frequency array.

    Returns
    -------
    ndarray :
        Symmetric distance array.

    """
    # todo: implement co-occurrence
    return np.eye(5)


def gravity(x: np.ndarray):
    r"""Calculate distance matrix from frequency using the gravity method.

    .. math::
        \delta_{ij} = \sqrt{\frac{\sum_{i}{f_{ij}}\sum_{j}{f_{ij}}}
        {f_{ij}\sum_{i,j}{f_{ij}}}}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, and :math:`f_{ij}` is their frequency.

    Parameters
    -----------
    x : ndarray
        Symmetric frequency array.

    Returns
    -------
    ndarray :
        Symmetric distance array.

    """
    # todo: implement gravity
    return np.eye(5)


def probability(x: np.ndarray):
    r"""Calculate distance matrix from probability matrix.

    .. math::
    \delta_{ij} = \sqrt{-\ln{\frac{s_{ij}}{max(s_{ij})}}}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, and :math:`p_{ij}` is their probablity.

    Parameters
    -----------
    x : ndarray
        Symmetric probability array.

    Returns
    -------
    ndarray :
        Symmetric distance array.

    """
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError(
            "There is an out of bounds value. Please make"
            "sure all correlations are between [0, 1]."
        )
    y = np.arcsin(x)
    dist = 1 / np.sqrt(y)
    return dist


def covariance(x: np.ndarray):
    r"""Calculate distance matrix from similarity using the covariance method.

    .. math::
    \delta_{ij} = \sqrt{s_{ii}+s_{jj}-2s_{ij}}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, :math:`s_{ii}` and :math:`s_{jj}` are the variances
    of feature :math:`i` and feature :math:`j`, and :math:`s_{ij}`
    is the covariance between the two features.

    Parameters
    -----------
    x : ndarray
        Symmetric covariance array.

    Returns
    -------
    ndarray :
        Symmetric distance array.

    """
    # todo: optimize performance if possible
    # initialize distance matrix
    dist = np.empty(x.shape)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            # for each entry in covariance matrix, calculate distance
            dist[i][j] = x[i][i] + x[j][j] - 2 * x[i][j]
    dist = np.sqrt(dist)
    return dist


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
