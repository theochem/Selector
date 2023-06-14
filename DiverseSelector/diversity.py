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

"""Molecule dataset diversity calculation module."""

from typing import List

import numpy as np
from scipy.spatial.distance import euclidean

__all__ = [
    "compute_diversity",
    "entropy",
    "logdet",
    "shannon_entropy",
    "wdud",
    "total_diversity_volume",
    "gini_coefficient",
]


def compute_diversity(
    features: np.array,
    div_type: str = "total_diversity_volume",
) -> float:
    """Compute diversity metrics.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    div_type : str, optional
        Method of calculation diversity for a given molecule set, which
        includes "entropy", "logdet", "shannon_entropy", "wdud",
        gini_coefficient" and "total_diversity_volume". Default is "total_diversity_volume".
    mols : List[rdkit.Chem.rdchem.Mol], optional
        List of RDKit molecule objects. This is only needed when using the
        "explicit_diversity_index" method. Default=None.

    Returns
    -------
    float, computed diversity.

    """
    func_dict = {
        "entropy": entropy,
        "logdet": logdet,
        "shannon_entropy": shannon_entropy,
        "wdud": wdud,
        "total_diversity_volume": total_diversity_volume,
        "gini_coefficient": gini_coefficient,
    }

    if div_type in func_dict:
        return func_dict[div_type](features)
    else:
        raise ValueError(f"Diversity type {div_type} not supported.")


def entropy(x: np.ndarray) -> float:
    r"""Compute entropy of matrix. The equation for entropy is

    .. math::
        E = $-\frac{\sum{\frac{y_i}{N}\ln{\frac{y_i}{N}}}}{L\frac{\ln{2}}{2}}$

    where N is the number of molecules in the set, L is the length of the fingerprint,
    and :math:y_i is a vector of the bitcounts of each feature in the fingerprints.

    Parameters
    ----------
    x : ndarray
        Feature matrix.

    Returns
    -------
    e : float
        Entropy of matrix.

    Notes
    -----
    Feature matrices are converted to bits,
    so we lose any information associated with num in matrix.
    Weidlich, I. E., and Filippov, I. V. (2016)
    Using the Gini coefficient to measure the chemical diversity of small-molecule libraries.
    Journal of Computational Chemistry 37, 2091-2097.
    """

    # convert input matrix to bit matrix
    y = np.empty(x.shape)
    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            if x[i, j] != 0:
                y[i, j] = 1
            else:
                y[i, j] = 0
    # initialize variables
    length = len(x[0])
    n = len(x)
    top = 0
    val = []
    for i in range(0, length):  # count bits in fingerprint
        val.append(sum(y[:, i]))
    ans = np.sort(val)
    # sum entropy calculation for each feature
    for i in range(0, length):
        if ans[i] == 0:
            raise ValueError
        if ans[i] != 0:
            top += ((ans[i]) / n) * (np.log(ans[i] / n))
    e = -1 * (top / (length * 0.34657359027997264))
    return e


def logdet(x: np.ndarray) -> float:
    r"""Computes the log determinant function.

     Input is an :math:S\times :math:n feature matrix with
    :math:S molecules and :math:n features.

    .. math:
        F_{logdet}\left(S\right) = \log{\det{\left(X[S]X[S]^T + I_{|S|} \right)}}

    Parameters
    ----------
    x : ndarray(S, n)
        Subset feature matrix.

    Returns
    -------
    f_logdet: float
        The volume of parallelotope spand by the matrix.

    Notes
    -----
    Nakamura, T., Sakaue, S., Fujii, K., Harabuchi, Y., Maeda, S., and Iwata, S.. (2022)
    Selecting molecules with diverse structures and properties by maximizing
    submodular functions of descriptors learned with graph neural networks.
    Scientific Reports 12.
    """
    mid = np.dot(x, np.transpose(x))
    f_logdet = np.log10(np.linalg.det(mid + np.identity(len(x))))
    return f_logdet


def shannon_entropy(x: np.ndarray) -> float:
    r"""Computes the shannon entropy of a matrix.

    The equation for Shannon entropy is

    .. math::
        H(X) = \sum_{i=1}^{n}-P_i(X)\log{P_i(X)}

    where X is the feature matrix, n is the number of features, and :math:`P_i(X)` is the
    proportion of the ith descriptor in X.

    Parameters
    ----------
    x : ndarray
        Bit-string matrix.

    Returns
    -------
    h_x: float
        The shannon entropy of the matrix.

    Notes
    -----
    Leguy, J., Glavatskikh, M., Cauchy, T., and Benoit. (2021)
    Scalable estimator of the diversity for de novo molecular generation resulting
    in a more robust QM dataset (OD9) and a more efficient molecular optimization.
    Journal of Cheminformatics 13.
    """
    size = len(x[:, 0])
    h_x = 0
    for i in range(0, size):
        # calculate feature proportion
        p_i = np.count_nonzero(x[:, i]) / size
        # sum all non-zero terms
        if p_i == 0:
            raise ValueError(f"Feature {i} has value 0 for all molecules. Remove extraneous feature from data set.")
        h_x += (-1 * p_i) * np.log10(p_i)
    return h_x


def wdud(x: np.ndarray) -> float:
    r"""Computes the Wasserstein Distance to Uniform Distribution(WDUD).
    The equation for the Wasserstein Distance is
    .. math::
        WDUD(x) = \int_{v_{min}}^{v_{max}} |U(x) - V(x)|dx

    where :math:v_{min} and :math:v_{max} are the minimum and maximum
    feature values in :math:x, and :math:U(x) is the uniform distribution between them.
    :math:V(x) is the discrete distribution of feature values in :math:x.

    Parameters
    ----------
    x : ndarray
        Feature matrix.

    Returns
    -------
    h_x: float
        The WDUD of the matrix.

    Notes
    -----
    Unclear if this method is implemented correctly.
    Nakamura, T., Sakaue, S., Fujii, K., Harabuchi, Y., Maeda, S., and Iwata, S.. (2022)
    Selecting molecules with diverse structures and properties by maximizing
    submodular functions of descriptors learned with graph neural networks.
    Scientific Reports 12.
    """
    # min_max normalization:
    d = len(x[0])
    n = len(x[:, 0])
    max_x = np.max(x)
    min_x = np.min(x)
    y = np.zeros(x.shape)
    for i in range(0, len(x[:, 0])):
        for j in range(0, len(x[0])):
            # value of uniform distribution at all data points
            y[i, j] = (x[i, j] - min_x) / (max_x - min_x)
    # wdud
    ans = []
    for i in range(0, d):
        h = -np.sort(-y[:, i])
        wdu = (-1 / d) - h[0]
        for j in range(1, len(h)):
            wdu -= np.absolute(((j - 1) / d) - h[j])
        ans.append(wdu)
    return np.average(ans)


def total_diversity_volume(x: np.ndarray) -> float:
    r"""Computes the total diversity volume of the matrix.

    .. math::


    Parameters
    ----------
    x : ndarray
        Feature matrix.

    Returns
    -------
    g_s: float
        The total diversity volume of the matrix.

    Notes
    -----
    Agrafiotis, D. K.. (1997) Stochastic Algorithms for Maximizing Molecular Diversity.
    Journal of Chemical Information and Computer Sciences 37, 841-851.
    """
    d = len(x[0])
    k = len(x[:, 0])
    # min_max normalization:
    max_x = max(map(max, x))
    min_x = min(map(min, x))
    y = np.zeros((k, d))
    for i in range(0, k):
        for j in range(0, d):  # scale data according to min-max distribution
            y[i, j] = (x[i, j] - min_x) / (max_x - min_x)
    # calculate diversity volume
    r_o = d * np.sqrt(1 / k)  # hypersphere radius
    g_s = 0
    for i in range(0, (k - 1)):
        for j in range((i + 1), k):
            dist = euclidean(y[i], y[j])
            # Overlap penalty
            if dist <= (2 * r_o) and dist != 0:
                o_ij = min(100, 2 * r_o / dist - 1)
                g_s += o_ij
            else:
                o_ij = 0
                g_s += o_ij
    return g_s


def gini_coefficient(a: np.ndarray):
    r"""
    Gini coefficient of bit-wise fingerprints of a database of molecules.

    Measures the chemical diversity of a database of molecules defined by
    the following formula:

    .. math::
        G = \frac{2 \sum_{i=1}^L i ||y_i||_1 }{N \sum_{i=1}^L ||y_i||_1} - \frac{L+1}{L},

    where :math:`y_i \in \{0, 1\}^N` is a vector of zero and ones of length the
    number of molecules :math:`N` of the `i`th feature, and :math:`L` is the feature length.

    Parameters
    ----------
    a : ndarray(N, L)
        Molecule features in L bits with N molecules.

    Returns
    -------
    float :
        Gini coefficient between zero and one, where closer to zero indicates more diversity.

    References
    ----------
    .. [1] Weidlich, Iwona E., and Igor V. Filippov. "Using the gini coefficient to measure the
           chemical diversity of small‐molecule libraries." (2016): 2091-2097.

    """
    # Check that `a` is a bit-wise fingerprint.
    if np.any(np.abs(np.sort(np.unique(a)) - np.array([0, 1])) > 1e-8):
        raise ValueError("Attribute `a` should have binary values.")
    if a.ndim != 2:
        raise ValueError(
            f"Attribute `a` should have dimension two rather than {a.ndim}."
        )

    numb_features = a.shape[1]
    # Take the bit-count of each column/molecule.
    bit_count = np.sum(a, axis=0)

    # Sort the bit-count since Gini coefficients relies on cumulative distribution.
    bit_count = np.sort(bit_count)

    # Mean of denominator
    denominator = numb_features * np.sum(bit_count)
    numerator = np.sum(np.arange(1, numb_features + 1) * bit_count)

    return 2.0 * numerator / denominator - (numb_features + 1) / numb_features
