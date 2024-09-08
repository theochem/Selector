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
"""Similarity Module."""

from itertools import combinations_with_replacement

import numpy as np

__all__ = [
    "pairwise_similarity_bit",
    "tanimoto",
    "modified_tanimoto",
    "scaled_similarity_matrix",
]


def pairwise_similarity_bit(X: np.array, metric: str) -> np.ndarray:
    """Compute pairwise similarity coefficient matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix of `n_samples` samples in `n_features` dimensional space.
    metric : str
        The metric used when calculating similarity coefficients between samples in a feature array.
        Method for calculating similarity coefficient. Options: `"tanimoto"`, `"modified_tanimoto"`.

    Returns
    -------
    s : ndarray of shape (n_samples, n_samples)
        A symmetric similarity matrix between each pair of samples in the feature matrix.
        The diagonal elements are directly computed instead of assuming that they are 1.

    """

    available_methods = {
        "tanimoto": tanimoto,
        "modified_tanimoto": modified_tanimoto,
    }
    if metric not in available_methods:
        raise ValueError(
            f"Argument metric={metric} is not recognized! Choose from {available_methods.keys()}"
        )
    if X.ndim != 2:
        raise ValueError(f"Argument features should be a 2D array, got {X.ndim}")

    # make pairwise m-by-m similarity matrix
    n_samples = len(X)
    s = np.zeros((n_samples, n_samples))
    # compute similarity between all pairs of points (including the diagonal elements)
    for i, j in combinations_with_replacement(range(n_samples), 2):
        s[i, j] = s[j, i] = available_methods[metric](X[i], X[j])
    return s


def tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute Tanimoto coefficient or index (a.k.a. Jaccard similarity coefficient).

    For two binary or non-binary arrays :math:`A` and :math:`B`, Tanimoto coefficient
    is defined as the size of their intersection divided by the size of their union:

    .. math::
        T(A, B) = \frac{| A \cap B|}{| A \cup B |} =
        \frac{| A \cap B|}{|A| + |B| - | A \cap B|} =
        \frac{A \cdot B}{\|A\|^2 + \|B\|^2 - A \cdot B}

    where :math:`A \cdot B = \sum_i{A_i B_i}` and :math:`\|A\|^2 = \sum_i{A_i^2}`.

    Parameters
    ----------
    a : ndarray of shape (n_features,)
        The 1D feature array of sample :math:`A` in an `n_features` dimensional space.
    b : ndarray of shape (n_features,)
        The 1D feature array of sample :math:`B` in an `n_features` dimensional space.

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
        raise ValueError(
            f"Arguments a and b should have the same shape, got {a.shape} != {b.shape}"
        )
    coeff = sum(a * b) / (sum(a**2) + sum(b**2) - sum(a * b))
    return coeff


def modified_tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute the modified tanimoto coefficient from bitstring vectors of data points A and B.

    Adjusts calculation of the Tanimoto coefficient to counter its natural bias towards
    shorter vectors using a Bernoulli probability model.

    .. math::
        {mt} = \frac{2-p}{3} T_1 + \frac{1+p}{3} T_0

    where :math:`p` is success probability of independent trials,
    :math:`T_1` is the number of common '1' bits between data points
    (:math:`T_1 = | A \cap B |`), and :math:`T_0` is the number of common '0'
    bits between data points (:math:`T_0 = |(1-A) \cap (1-B)|`).


    Parameters
    ----------
    a : ndarray of shape (n_features,)
        The 1D bitstring feature array of sample :math:`A` in an `n_features` dimensional space.
    b : ndarray of shape (n_features,)
        The 1D bitstring feature array of sample :math:`B` in an `n_features` dimensional space.

    Returns
    -------
    mt : float
        Modified tanimoto coefficient between bitstring feature arrays :math:`A` and :math:`B`.

    Notes
    -----
    The equation above has been derived from

    .. math::
       {mt}_{\alpha} = {\alpha}T_1 + (1-\alpha)T_0

    where :math:`\alpha = \frac{2-p}{3}`. This is done so that the expected value
    of the modified tanimoto, :math:`E(mt)`, remains constant even as the number of
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
    if a.shape != b.shape:
        raise ValueError(
            f"Arguments a and b should have the same shape, got {a.shape} != {b.shape}"
        )

    n_features = len(a)
    # number of common '1' bits between points A and B
    n_11 = sum(a * b)
    # number of common '0' bits between points A and B
    n_00 = sum((1 - a) * (1 - b))

    # calculate Tanimoto coefficient based on '0' bits
    t_1 = 1
    if n_00 != n_features:
        # bit strings are not all '0's
        t_1 = n_11 / (n_features - n_00)
    # calculate Tanimoto coefficient based on '1' bits
    t_0 = 1
    if n_11 != n_features:
        # bit strings are not all '1's
        t_0 = n_00 / (n_features - n_11)

    # combine into modified tanimoto using Bernoulli Model
    # p = independent success trials
    #       evaluated as total number of '1' bits
    #       divided by 2x the fingerprint length
    p = (n_features - n_00 + n_11) / (2 * n_features)
    # mt = x * T_1 + (1-x) * T_0
    #       x = (2-p)/3 so that E(mt) = 1/3, no matter the value of p
    mt = (((2 - p) / 3) * t_1) + (((1 + p) / 3) * t_0)
    return mt


def scaled_similarity_matrix(X: np.array) -> np.ndarray:
    r"""Compute the scaled similarity matrix.

    .. math::
        X(i,j) = \frac{X(i,j)}{\sqrt{X(i,i)X(j,j)}}

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_samples)
        Similarity matrix of `n_samples`.

    Returns
    -------
    s : ndarray of shape (n_samples, n_samples)
        A scaled symmetric similarity matrix.

    """
    if X.ndim != 2:
        raise ValueError(f"Argument similarity matrix should be a 2D array, got {X.ndim}")

    if X.shape[0] != X.shape[1]:
        raise ValueError(
            f"Argument similarity matrix should be a square matrix (having same number of rows and columns), got {X.shape[0]} and {X.shape[1]}"
        )

    if not (np.all(X >= 0) and np.all(np.diag(X) > 0)):
        raise ValueError(
            "All elements of similarity matrix should be greater than zero and diagonals should be non-zero"
        )

    # scaling does not happen if the matrix is binary similarity matrix with all diagonal elements as 1
    if np.all(np.diag(X) == 1):
        print("No scaling is taking effect")
        return X
    else:
        # make a scaled similarity matrix
        n_samples = len(X)
        s = np.zeros((n_samples, n_samples))
        # calculate the square root of the diagonal elements
        sqrt_diag = np.sqrt(np.diag(X))
        # calculate the product of the square roots of the diagonal elements
        product_sqrt_diag = np.outer(sqrt_diag, sqrt_diag)
        # divide each element of the matrix by the product of the square roots of diagonal elements
        s = X / product_sqrt_diag
        return s


def similarity_index(x: np.array, y: np.array, sim_index: str) -> float:
    """Compute similarity index matrix.

    Parameters
    ----------
    x : ndarray of shape (n_features,)
        Feature array of sample `x` in an `n_features` dimensional space
    y : ndarray of shape (n_features,)
        Feature array of sample `y` in an `n_features` dimensional space
    sim_index : str, optional
        The key with the abbreviation of the similarity index to be used for calculations.
        Possible values are:
            - 'AC': Austin-Colwell
            - 'BUB': Baroni-Urbani-Buser
            - 'CTn': Consoni-Todschini n (n=1,2)
            - 'Fai': Faith
            - 'Gle': Gleason
            - 'Ja': Jaccard
            - 'JT': Jaccard-Tanimoto
            - 'RT': Rogers-Tanimoto
            - 'RR': Russel-Rao
            - 'SM': Sokal-Michener
            - 'SSn': Sokal-Sneath n (n=1,2)
        Default is 'RR'.

    Returns
    -------
    sim : float
        The similarity index value between the feature arrays `x` and `y`.
    """
    # Define the similarity index functions
    similarity_indices = {
        "AC": lambda a, d, dis, p: 2 / np.pi * np.arcsin(((a + d) / p) ** 0.5),
        "BUB": lambda a, d, dis, p: ((a * d) ** 0.5 + a) / ((a * d) ** 0.5 + a + dis),
        "CT1": lambda a, d, dis, p: np.log(1 + a + d) / np.log(1 + p),
        "CT2": lambda a, d, dis, p: (np.log(1 + p) - np.log(1 + dis)) / np.log(1 + p),
        "Fai": lambda a, d, dis, p: (a + 0.5 * d) / p,
        "Gle": lambda a, d, dis, p: 2 * a / (2 * a + dis),
        "Ja": lambda a, d, dis, p: 3 * a / (3 * a + dis),
        "JT": lambda a, d, dis, p: a / (a + dis),
        "RT": lambda a, d, dis, p: (a + d) / (p + dis),
        "RR": lambda a, d, dis, p: a / p,
        "SM": lambda a, d, dis, p: (a + d) / p,
        "SS1": lambda a, d, dis, p: a / (a + 2 * dis),
        "SS2": lambda a, d, dis, p: (2 * (a + d)) / (p + (a + d)),
    }

    if sim_index not in similarity_indices:
        raise ValueError(
            f"Argument sim_index={sim_index} is not recognized! Choose from {similarity_indices.keys()}"
        )
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"Arguments x and y should be 1D arrays, got {x.ndim} and {y.ndim}")
    if x.shape != y.shape:
        raise ValueError(
            f"Arguments x and y should have the same shape, got {x.shape} != {y.shape}"
        )
    a, d, dis, p = _compute_base_descriptors(x, y)
    return similarity_indices[sim_index](a, d, dis, p)


def _compute_base_descriptors(x, y):
    """Compute the base descriptors for the similarity indices.

    Parameters
    ----------
    x : ndarray of shape (n_features,)
        Feature array of sample `x` in an `n_features` dimensional space
    y : ndarray of shape (n_features,)
        Feature array of sample `y` in an `n_features` dimensional space

    Returns
    -------
    tuple(int, int, int, int)
        The number of common on bits, number of common off bits, number of 1-0 mismatches, and the
        length of the fingerprint.
    """
    p = len(x)
    a = np.dot(x, y)
    d = np.dot(1 - x, 1 - y)
    dis = p - a - d
    return a, d, dis, p
