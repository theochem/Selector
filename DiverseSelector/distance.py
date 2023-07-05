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
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform

__all__ = [
    "pairwise_similarity_bit",
    "tanimoto",
    "modified_tanimoto",
    "nearest_average_tanimoto",
]


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
        Similarity coefficients for all data point pairs in feature matrix.
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
    # shape into symmetric matrix
    pair_coeff = squareform(pair_simi) + np.identity(size)
    return pair_coeff


def tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute Tanimoto coefficient.

    ..math::
        T(A,B) = A \cap B / A \cup B

    Parameters
    ----------
    a : array_like
        Data point A's features.
    b : array_like
        Data point B's features.

    Returns
    -------
    coeff : float
        Tanimoto coefficient for data points A and B.

    Notes
    -----
    The Tanimoto coefficient computes similarity by taking the intersection
    of A and B over their union.

    Bajusz, D., Rácz, A., and Héberger, K.. (2015)
    Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?.
    Journal of Cheminformatics 7.
    """
    coeff = (sum(a * b)) / ((sum(a**2)) + (sum(b**2)) - (sum(a * b)))
    return coeff


def modified_tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute the modified tanimoto coefficient from bitstring vectors of data points A and B.

    Adjusts calculation of the Tanimoto coefficient to counter its natural bias towards
    shorter vectors using a Bernoulli probability model.

    ..math::
    MT = \frac{2-p}{3}T_1 + \frac{1+p}{3}T_0

    where :math:`p` is success probability of independent trials,
    :math:`T_1` is the number of common '1' bits between data points
    (:math:`T_1 = | A \cap B |`), and :math:`T_0` is the number of common '0'
    bits between data points (:math:`T_0 = |(1-A) \cap (1-B)|`).


    Parameters
    ----------
    a : array_like
        Data point A's features in bitstring.
    b : array_like
        Data point B's features in bitstring.

    Returns
    -------
    mt : float
        Modified tanimoto coefficient for molecule A and B.

    Notes
    -----
    The equation above has been derived from

    ..math::
    MT_\alpha= {\alpha}T_1 + (1-\alpha)T_0

    where :math:`\alpha = \frac{2-p}{3}`. This is done so that the expected value
    of the modified tanimoto, :math:`E(MT)`, remains constant even as the number of
    trials :math:`p` grows larger.

    Fligner, M. A., Verducci, J. S., and Blower, P. E.. (2002)
    A Modification of the Jaccard-Tanimoto Similarity Index for
    Diverse Selection of Chemical Compounds Using Binary Strings.
    Technometrics 44, 110-119.
    """
    if a.ndim != 1:
        raise ValueError(f"Argument `a` should have dimension 1 rather than {a.ndim}.")
    if b.ndim != 1:
        raise ValueError(f"Argument `b` should have dimension 1 rather than {b.ndim}.")

    n = len(a)
    # number of common '1' bits between points A and B
    n_11 = sum(a * b)
    # number of common '0' bits between points A and B
    n_00 = sum((1 - a) * (1 - b))

    # calculate Tanimoto coeff based on '1' bits
    if n_00 == n:
        # bit string is all '0's
        t_1 = 1
    else:
        t_1 = n_11 / (n - n_00)
    # calculate Tanimoto coeff based on '1' bits
    if n_11 == n:
        # bit string is all '1's
        t_0 = 1
    else:
        t_0 = n_00 / (n - n_11)
    # combine into modified tanimoto using Bernoulli Model
    # p = independent success trials
    #       evaluated as total number of '1' bits
    #       divided by 2x the fingerprint length
    p = (n - n_00 + n_11) / (2 * n)
    # mt = x * T_1 + (1-x) * T_0
    #       x = (2-p)/3 so that E(mt) = 1/3, no matter the value of p
    mt = (((2 - p) / 3) * t_1) + (((1 + p) / 3) * t_0)
    return mt


def nearest_average_tanimoto(x: np.ndarray) -> float:
    """Computes the average tanimoto for nearest data points.

    Parameters
    ----------
    x : ndarray
        Feature matrix.

    Returns
    -------
    float :
        Average Tanimoto of closest pairs.

    Notes
    -----
    This computes the Tanimoto coefficient of pairs of data points
    with the shortest distances, then returns the average of them.
    This calculation is explicitly for the explicit diversity index.

    Papp, Á., Gulyás-Forró, A., Gulyás, Z., Dormán, G., Ürge, L.,
    and Darvas, F.. (2006) Explicit Diversity Index (EDI):
    A Novel Measure for Assessing the Diversity of Compound Databases.
    Journal of Chemical Information and Modeling 46, 1898-1904.
    """
    tani = []
    # calculate euclidean distance between all points
    #     and adjust for distance to self
    dist = distance_matrix(x, x) + np.inf*np.eye(x.shape[0])
    # find closest point for each row of x
    short_idx = np.argmin(dist, axis=0)
    for idx, min_d in enumerate(short_idx):
        # compute the tanimoto coeff for each pair of closest points
        tani.append(tanimoto(x[idx], x[min_d]))
    # take the average of all coeffs calculated
    nat = np.average(tani)
    return nat
