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

"""Metric calculation module."""


import numpy as np
from scipy.spatial.distance import squareform

__all__ = [
    "compute_distance_matrix",
    "pairwise_similarity_bit",
    "tanimoto",
    "modified_tanimoto",
    "nearest_average_tanimoto"
]


def compute_distance_matrix(
    features: np.ndarray,
    metric: str
):
    """Compute pairwise distance given a feature matrix.

    Parameters
    ----------
    features : np.ndarray
        Molecule feature matrix.
    metric : str
        Distance metric.

    Returns
    -------
    dist : ndarray
        Symmetric distance array.
    """
    # todo: add more metrics implemented here
    built_in_metrics = {
        "tanimoto": tanimoto,
        "modified_tanimoto": modified_tanimoto,
    }

    # Check if specified metric is supported
    if metric in built_in_metrics:
        distances = []
        size = len(features)
        for i in range(0, size):
            for j in range(i + 1, size):
                # use the metric to compute distance between all molecule pairs
                distances.append(1 - built_in_metrics[metric](features[i], features[j]))
        dist = squareform(distances)  # shape into symmetric matrix

    else:  # raise error if unsupported
        raise ValueError(f"Metric {metric} is not supported by the library.")

    return dist


def pairwise_similarity_bit(features: np.array, metric: str) -> np.ndarray:
    """Compute the pairwise similarity coefficients and returns them in
        a square symmetric matrix.

    Parameters
    ----------
    features : ndarray
        Feature matrix.
    metric : str
        Method of calculation.

    Returns
    -------
    pair_coeff : ndarray
        Similarity coefficients for all molecule pairs in feature matrix.
    """

    function_dict = {
        "tanimoto": tanimoto,
        "modified_tanimoto": modified_tanimoto,
    }

    pair_simi = []
    size = len(features)
    for i in range(0, size):
        for j in range(i + 1, size):
            # use the specified metric to compute similarity between all distinct molecule pairs
            pair_simi.append(function_dict[metric](features[i], features[j]))
    pair_coeff = squareform(pair_simi) + np.identity(size)  # shape into symmetric matrix
    return pair_coeff


def tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute Tanimoto coefficient.

    ..math::
        T(A,B) = A \cap B / A \cup B

    Parameters
    ----------
    a : array_like
        Molecule A's features.
    b : array_like
        Molecules B's features.

    Returns
    -------
    coeff : float
        Tanimoto coefficient for molecules A and B.

    Notes
    -----
    The Tanimoto coefficient computes similarity by taking the intersection of A and B over their union.

    Bajusz, D., Rácz, A., and Héberger, K.. (2015)
    Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?.
    Journal of Cheminformatics 7.
    """
    coeff = (sum(a * b)) / ((sum(a ** 2)) + (sum(b ** 2)) - (sum(a * b)))
    return coeff


def modified_tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute the modified tanimoto coefficient from bitstrings of molecules A and B.

    Adjusts calculation of the Tanimoto coefficient to counter its natural bias towards
    smaller molecules using a Bernoulli probability model.

    ..math::
    mt = \frac{2-p}{3}t_1 + \frac{1+p}{3}t_0$
    where
    p = success probability of independent trials
    $t_1 = | A \cap B |$
    $t_0 =  |(1-A) \cap (1-B)|$

    Parameters
    ----------
    a : array_like
        Molecule A's features in bitstring.
    b : array_like
        Molecules B's features in bitstring.

    Returns
    -------
    mt : float
        Modified tanimoto coefficient for molecule A and B.

    Notes
    -----

    Fligner, M. A., Verducci, J. S., and Blower, P. E.. (2002)
    A Modification of the Jaccard-Tanimoto Similarity Index for
    Diverse Selection of Chemical Compounds Using Binary Strings.
    Technometrics 44, 110-119.
    """
    n = len(a)
    # intersection of '1' bits
    n_11 = sum(a * b)
    # intersection of '0' bits
    n_00 = sum((1 - a) * (1 - b))

    # calculate in terms of '1' bits
    if n_00 == n:
        t_1 = 1
    else:
        t_1 = n_11 / (n - n_00)
    # calculate in terms of '0' bits
    if n_11 == n:
        t_0 = 1
    else:
        t_0 = n_00 / (n - n_11)
    # combine into modified tanimoto using Bernoulli Model
    p = ((n - n_00) + n_11) / (2 * n)
    mt = (((2 - p) / 3) * t_1) + (((1 + p) / 3) * t_0)
    return mt


def nearest_average_tanimoto(x: np.ndarray) -> float:
    """Computes the average tanimoto for nearest molecules.

    Parameters
    ----------
    x : ndarray
        Feature matrix.

    Returns
    -------
    nat : float
        Average tanimoto of closest pairs.

    Notes
    -----
    This computes the tanimoto coefficient of pairs with the shortest
    distances, then returns the average of them.
    This calculation is explictly for the explicit diversity index.

    Papp, Á., Gulyás-Forró, A., Gulyás, Z., Dormán, G., Ürge, L.,
    and Darvas, F.. (2006) Explicit Diversity Index (EDI):
    A Novel Measure for Assessing the Diversity of Compound Databases.
    Journal of Chemical Information and Modeling 46, 1898-1904.
    """
    tani = []
    for idx, _ in enumerate(x):
        short = 100  # arbitrary distance
        a = 0
        b = 0
        for jdx, _ in enumerate(x):  # search for shortest distance point from idx
            dist = np.linalg.norm(x[idx]-x[jdx])
            if dist < short and idx != jdx:
                short = dist
                a = idx
                b = jdx
        # calculate tanimoto for each shortest dist pair
        tani.append(tanimoto(x[a], x[b]))
    # compute average of all shortest tanimoto coeffs
    nat = np.average(tani)
    return nat

