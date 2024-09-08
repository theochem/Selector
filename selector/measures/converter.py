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
"""Module for converting similarity measures to distance/dissimilarity measures."""

from typing import Union

import numpy as np

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


def sim_to_dist(
    x: Union[int, float, np.ndarray], metric: str, scaling_factor: float = 1.0
) -> Union[float, np.ndarray]:
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
    scaling_factor : float, optional
        Scaling factor for the distance array. Default is 1.0.

    Returns
    -------
    dist : float or ndarray
         Distance value or array.
    """
    # scale the distance matrix
    x = x * scaling_factor

    frequency = {
        "transition": transition,
        "co-occurrence": co_occurrence,
        "gravity": gravity,
    }
    method_dict = {
        "reverse": reverse,
        "reciprocal": reciprocal,
        "exponential": exponential,
        "gaussian": gaussian,
        "correlation": correlation,
        "probability": probability,
        "covariance": covariance,
    }

    # check if x is a single value
    single_value = False
    if isinstance(x, (float, int)):
        x = np.array([[x]])
        single_value = True

    # check x
    if not isinstance(x, np.ndarray):
        raise ValueError(f"Argument x should be a numpy array instead of {type(x)}")
    # check that x is a valid array
    if x.ndim != 1 and x.ndim != 2:
        raise ValueError(f"Argument x should either have 1 or 2 dimensions, got {x.ndim}.")
    if x.ndim == 1 and metric in ["co-occurrence", "gravity"]:
        raise ValueError(f"Argument x should be a 2D array when using the {metric} metric.")
    # check if x is symmetric
    if x.ndim == 2 and not np.allclose(x, x.T):
        raise ValueError("Argument x should be a symmetric array.")

    # call correct metric function
    if metric in frequency:
        if np.any(x <= 0):
            raise ValueError(
                "There is a negative or zero value in the input. Please "
                "make sure all frequency values are positive."
            )
        dist = frequency[metric](x)
    elif metric in method_dict:
        dist = method_dict[metric](x)
    elif metric == "membership" or metric == "confusion":
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError(
                "There is an out of bounds value. Please make "
                "sure all input values are between [0, 1]."
            )
        dist = 1 - x
    # unsupported metric
    else:
        raise ValueError(f"{metric} is an unsupported metric.")

    # convert back to float if input was single value
    if single_value:
        dist = dist.item((0, 0))

    return dist


def reverse(x: np.ndarray) -> np.ndarray:
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
    dist : ndarray
        Distance array.

    """
    dist = np.max(x) + np.min(x) - x
    return dist


def reciprocal(x: np.ndarray) -> np.ndarray:
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
    dist : ndarray
        Distance array.
    """

    if np.any(x <= 0):
        raise ValueError(
            "There is an out of bounds value. Please make " "sure all similarities are positive."
        )
    return 1 / x


def exponential(x: np.ndarray) -> np.ndarray:
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
    dist : ndarray
        Distance array.

    """
    max_sim = np.max(x)
    if max_sim == 0:
        raise ValueError("Maximum similarity in `x` is 0. Distance cannot be computed.")
    dist = -np.log(x / max_sim)
    return dist


def gaussian(x: np.ndarray) -> np.ndarray:
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
    dist : ndarray
        Distance array.

    """
    max_sim = np.max(x)
    if max_sim == 0:
        raise ValueError("Maximum similarity in `x` is 0. Distance cannot be computed.")
    y = x / max_sim
    dist = np.sqrt(-np.log(y))
    return dist


def correlation(x: np.ndarray) -> np.ndarray:
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
    dist : ndarray
        Distance array.

    """
    if np.any(x < -1) or np.any(x > 1):
        raise ValueError(
            "There is an out of bounds value. Please make "
            "sure all correlations are between [-1, 1]."
        )
    dist = np.sqrt(1 - x)
    return dist


def transition(x: np.ndarray) -> np.ndarray:
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
    dist : ndarray
        Distance array.

    """
    dist = 1 / np.sqrt(x)
    return dist


def co_occurrence(x: np.ndarray) -> np.ndarray:
    r"""Calculate distance array from frequency using the co-occurrence method.

    .. math::
        \delta_{ij} =  \left(1 + \frac{f_{ij}\sum_{i,j}{f_{ij}}}{\sum_{i}{f_{ij}}\sum_{j}{f_{ij}}} \right)^{-1}

    where :math:`\delta_{ij}` is the distance between points :math:`i`
    and :math:`j`, and :math:`f_{ij}` is their frequency.

    Parameters
    -----------
    x : ndarray
        Frequency array.

    Returns
    -------
    dist : ndarray
        Co-occurrence array.

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


def gravity(x: np.ndarray) -> np.ndarray:
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
    dist : ndarray
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


def probability(x: np.ndarray) -> np.ndarray:
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
    dist : ndarray
        Distance array.

    """
    if np.any(x <= 0) or np.any(x > 1):
        raise ValueError(
            "There is an out of bounds value. Please make "
            "sure all probabilities are between (0, 1]."
        )
    y = np.arcsin(x)
    dist = 1 / np.sqrt(y)
    return dist


def covariance(x: np.ndarray) -> np.ndarray:
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
    dist : ndarray
        Distance array.

    """
    variance = np.diag(x).reshape([x.shape[0], 1]) * np.ones([1, x.shape[0]])
    if np.any(variance < 0):
        raise ValueError("Variance of a single variable cannot be negative.")

    dist = variance + variance.T - 2 * x
    dist = np.sqrt(dist)
    return dist
