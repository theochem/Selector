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
from scipy.spatial.distance import squareform


__all__ = [
    "sim_to_dist",
    "reverse",
    "reciprocal",
    "exponential",
    "gaussian",
    "correlation",
    "transition",
    "co_occurrence",
    "gravity",
    "probability",
    "covariance",
]

# todo: figure out how to handle divide by zero errors
# affected functions: reciprocal,
#                     exponential,
#                     gaussian,
#                     transition,
#                     gravity,
#                     probability


def sim_to_dist(x, metric: str):
    """Convert similarity coefficients to distance array.

    Parameters
    ----------
    x : float or ndarray
        A similarity value as float, or a 1D or 2D array of similarity values.
        If 2D, the array is assumed to be symmetric.
    metric : str
        String or integer specifying which conversion metric to use.
        Supported metrics are "reverse", "reciprocal", "exponential",
        "gaussian", "membership", "correlation", "transition", "co-occurrence",
        "gravity", "confusion", "probability", and "covariance".

    Returns
    -------
    dist : float, ndarray
         Distance value or array.
    """
    method_dict = {
        "reverse": reverse,
        "reciprocal": reciprocal,
        "exponential": exponential,
        "gaussian": gaussian,
        "correlation": correlation,
        "transition": transition,
        "co-occurrence": co_occurrence,
        "gravity": gravity,
        "probability": probability,
        "covariance": covariance,
    }

    # check if x is a single value
    single = False
    if type(x) == float or type(x) == int:
        x = np.array([[x]])
        single = True

    # check if x is 1 array with invalid metric
    if x.ndim == 1:
        if metric == "co-occurrence" or metric == "gravity":
            raise RuntimeError(
                "This operation unsupported with 1D array. "
                "Please make sure `x` is a 2D array when using the metric ",
                metric,
            )

    # check that x is a valid array
    if x.ndim != 1 and x.ndim != 2:
        raise ValueError(
            f"Attribute `x` should have dimension 1 or 2 rather than {x.ndim}."
        )

    # call correct metric function
    dist = None
    if type(metric) == str:
        if metric in method_dict:
            dist = method_dict[metric](x)
        elif metric == "membership" or metric == "confusion":
            if np.any(x < 0) or np.any(x > 1):
                raise ValueError(
                    "There is an out of bounds value. Please make"
                    "sure all input values are between [0, 1]."
                )
            dist = 1 - x
    # unsupported metric
    else:
        raise ValueError(f"{metric} is an unsupported metric.")

    # convert back to float if input was single value
    if single:
        dist = dist.item((0, 0))

    return dist


def reverse(x: np.ndarray):
    r"""Calculate distance array from similarity using the reverse method.

    .. math::
    \delta_{ij} = min(s_{ij}) + max(s_{ij}) - s_{ij}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, :math:`s_{ij}` is their similarity coefficient,
    and :math:`max` and :math:`min` are the maximum and minimum
    values across the entire similarity array.

    Parameters
    -----------
    x : ndarray
        Similarity array.

    Returns
    -------
    ndarray :
        Distance array.

    """
    dist = (np.max(x) + np.min(x)) - x
    return dist


def reciprocal(x: np.ndarray):
    r"""Calculate distance array from similarity using the reciprocal method.

    .. math::
    \delta_{ij} = \frac{1}{s_{ij}}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, and :math:`s_{ij}` is their similarity coefficient.

    Parameters
    -----------
    x : ndarray
        Similarity array.

    Returns
    -------
    ndarray :
        Distance array.
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
        Similarity array.

    Returns
    -------
    ndarray :
        Distance array.

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
        Similarity array.

    Returns
    -------
    ndarray :
        Distance array.

    """
    y = x / (np.max(x))
    dist = np.sqrt(-np.log(y))
    return dist


def correlation(x: np.ndarray):
    r"""Calculate distance array from correlation array.

    .. math::
    \delta_{ij} = \sqrt{1 - r_{ij}}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, and :math:`r_{ij}` is their correlation.

    Parameters
    -----------
    x : ndarray
        Correlation array.

    Returns
    -------
    ndarray :
        Distance array.

    """
    if np.any(x < -1) or np.any(x > 1):
        raise ValueError(
            "There is an out of bounds value. Please make"
            "sure all correlations are between [-1, 1]."
        )
    dist = np.sqrt(1 - x)
    return dist


def transition(x: np.ndarray):
    r"""Calculate distance array from frequency using the transition method.

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
        Distance array.

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


def co_occurrence(x: np.ndarray):
    r"""Calculate distance array from frequency using the co-occurrence method.

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
        Symmetric co-occurence array.

    """
    # compute sums along each axis
    i = np.sum(x, axis=0, keepdims=True)
    j = np.sum(x, axis=1, keepdims=True)
    # multiply sums to scalar value
    bottom = np.dot(i, j)
    # multiply each element by the sum of entire array
    top = x * np.sum(x)
    # evaluate function as a whole
    dist = (1 + (top / bottom)) ** -1
    return dist


def gravity(x: np.ndarray):
    r"""Calculate distance array from frequency using the gravity method.

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
        Symmetric gravity array.

    """
    # compute sums along each axis
    i = np.sum(x, axis=0, keepdims=True)
    j = np.sum(x, axis=1, keepdims=True)
    # multiply sums to scalar value
    top = np.dot(i, j)
    # multiply each element by the sum of entire array
    bottom = x * np.sum(x)
    # take square root of the fraction
    dist = np.sqrt(top / bottom)
    return dist


def probability(x: np.ndarray):
    r"""Calculate distance array from probability array.

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
        Distance array.

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
    r"""Calculate distance array from similarity using the covariance method.

    .. math::
    \delta_{ij} = \sqrt{s_{ii}+s_{jj}-2s_{ij}}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, :math:`s_{ii}` and :math:`s_{jj}` are the variances
    of feature :math:`i` and feature :math:`j`, and :math:`s_{ij}`
    is the covariance between the two features.

    Parameters
    -----------
    x : ndarray
        Covariance array.

    Returns
    -------
    ndarray :
        Distance array.

    """
    # initialize distance array
    dist = np.empty(x.shape)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            # for each entry in covariance array, calculate distance
            dist[i][j] = x[i][i] + x[j][j] - 2 * x[i][j]
    dist = np.sqrt(dist)
    return dist
