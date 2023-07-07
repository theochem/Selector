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
from itertools import combinations_with_replacement
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform

__all__ = [
    "pairwise_similarity_bit",
    "tanimoto",
    "modified_tanimoto",
    "nearest_average_tanimoto",
]


def pairwise_similarity_bit(X: np.array, metric: str) -> np.ndarray:
    """Compute pairwise similarity coefficient matrix.

    Parameters
    ----------
    X : ndarray
        An `m` by `n` feature array of `m` samples in an `n`-dimensional feature space.
    metric : str
        Method for calculating similarity coefficient. Options: `"tanimoto"`, `"modified_tanimoto"`.

    Returns
    -------
    pair_simi : ndarray
        Returns a symmetric `m` by `m` array containing the similarity coefficient between
        each pair of samples in the feature matrix. The diagonal elements are directly
        computed instead of assuming that they are 1.
    """

    available_methods = {
        "tanimoto": tanimoto,
        "modified_tanimoto": modified_tanimoto,
    }
    if metric not in available_methods:
        raise ValueError(f"Argument metric={metric} is not recognized! Choose from {available_methods.keys()}")
    if X.ndim != 2:
        raise ValueError(f"Argument features should be a 2D array, got {X.ndim}")

    # make pairwise m-by-m similarity matrix
    m = len(X)
    pair_simi = np.zeros((m, m))
    # compute similarity between all pairs of points (including the diagonal elements)
    for i, j in combinations_with_replacement(range(m), 2):
        pair_simi[i, j] = pair_simi[j, i] = available_methods[metric](X[i], X[j])
    return pair_simi


def tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute Tanimoto coefficient or index (a.k.a. Jaccard similarity coefficient).

    For two binary or non-binary arrays :math:`A` and :math:`B`, Tanimoto coefficient
    is defined as the size of their intersection divided by the size of their union:

    ..math::
        T(A, B) = \frac{| A \cap B|}{| A \cup B |} =
        \frac{| A \cap B|}{|A| + |B| - | A \cap B|} =
        \frac{A \cdot B}{\|A\|^2 + \|B\|^2 - A \cdot B}

    where :math:`A \cdot B = \sum_i{A_i B_i}` and :math:`\|A\|^2 = \sum_i{A_i^2}`.

    Parameters
    ----------
    a : ndarray
        The 1D feature array of sample :math:`A` in an `n`-dimensional space.
    b : ndarray
        The 1D feature array of sample :math:`B` in an `n`-dimensional space.

    Returns
    -------
    coeff : float
        Tanimoto coefficient between feature arrays :math:`A` and :math:`B`.


    Bajusz, D., Rácz, A., and Héberger, K.. (2015)
    Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?.
    Journal of Cheminformatics 7.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Arguments a and b should be 1D arrays, got {a.ndim} and {b.ndim}")
    if a.shape != b.shape:
        raise ValueError(f"Arguments a and b should have the same shape, got {a.shape} != {b.shape}")
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


def nearest_average_tanimoto(X: np.ndarray) -> float:
    """Compute the average tanimoto for nearest data points measured by Minkowski 2-norm.

    For each sample, the closest neighbor is identified by computing its Minkowski 2-norm
    (i.e., Euclidean) distance with all other samples, and identifying neighboring sample
    with the shortest distance.

    Parameters
    ----------
    X : (M, K) array_like
        Matrix of `M` samples in an `K` dimensional feature space.

    Returns
    -------
    float :
        Average of the Tanimoto coefficients for each sample and its closest neighbor.

    Papp, Á., Gulyás-Forró, A., Gulyás, Z., Dormán, G., Ürge, L.,
    and Darvas, F.. (2006) Explicit Diversity Index (EDI):
    A Novel Measure for Assessing the Diversity of Compound Databases.
    Journal of Chemical Information and Modeling 46, 1898-1904.
    """
    # compute euclidean distance between all samples
    dist = distance_matrix(X, X, p=2)
    # replace zero self-distance with infinity, before computing nearest neighbors
    np.fill_diagonal(dist, np.inf)
    # find index of closest neighbor for each sample
    nearest_neighbors = np.argmin(dist, axis=0)
    assert nearest_neighbors.shape == (X.shape[0],)
    # compute the tanimoto coeff for each sample and its closest neighbor
    coeffs = []
    for idx_sample, idx_neighbor in enumerate(nearest_neighbors):
        coeffs.append(tanimoto(X[idx_sample], X[idx_neighbor]))
    # return average of all coefficients
    return np.average(coeffs)
