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
import warnings

__all__ = [
    "compute_diversity",
    "entropy",
    "logdet",
    "shannon_entropy",
    "wdud",
    "hypersphere_overlap_of_subset",
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
    r"""Compute entropy of matrix.

    The equation for entropy is
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
    # count bits in fingerprint
    for i in range(0, length):
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
    proportion of molecules that have feature :math:i in :math:X.

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
    num_feat = len(x[0, :])
    num_mols = len(x[:, 0])
    h_x = 0
    for i in range(0, num_feat):
        # calculate feature proportion
        p_i = np.count_nonzero(x[:, i]) / num_mols
        # sum all non-zero terms
        if p_i == 0:
            raise ValueError(f"Feature {i} has value 0 for all molecules. Remove extraneous feature from data set.")
        h_x += (-1 * p_i) * np.log10(p_i)
    return h_x


def wdud(x: np.ndarray) -> float:
    r"""Computes the Wasserstein Distance to Uniform Distribution(WDUD).

    The equation for the Wasserstein Distance for a single feature to uniform distribution is
    .. math::
        WDUD(x) = \int_{0}^{1} |U(x) - V(x)|dx

    where the feature is normalized to [0, 1], :math:`U(x)=x` is the cumulative distribution
    of the uniform distribution on [0, 1], and :math:`V(x) = \sum_{y <= x}1 / N` is the discrete
    distribution of the values of the feature in :math:`x`, where :math:`y` is the ith feature. This
    integral is calculated iteratively between :math:y_i and :math:y_{i+1}, using trapezoidal method.

    Parameters
    ----------
    x : ndarray(N, K)
        Feature array of N molecules and K features.

    Returns
    -------
    float:
        The mean of the WDUD of each feature over all molecules.

    Notes
    -----
    Nakamura, T., Sakaue, S., Fujii, K., Harabuchi, Y., Maeda, S., and Iwata, S.. (2022)
    Selecting molecules with diverse structures and properties by maximizing
    submodular functions of descriptors learned with graph neural networks.
    Scientific Reports 12.

    """
    if x.ndim != 2:
        raise ValueError(f"The number of dimensions {x.ndim} should be two.")
    # min_max normalization:
    num_features = len(x[0])
    num_mols = len(x[:, 0])
    # Find the maximum and minimum over each feature across all molecules.
    max_x = np.max(x, axis=0)
    min_x = np.min(x, axis=0)
    # Normalization of each feature to [0, 1]
    if np.any(np.abs(max_x - min_x) < 1e-30):
        raise ValueError(f"One of the features is redundant and causes normalization to fail.")
    x_norm = (x - min_x) / (max_x - min_x)
    ans = []  # store the Wasserstein distance for each feature
    for i in range(0, num_features):
        wdu = 0.0
        y = np.sort(x_norm[:, i])
        # Round to the sixth decimal place and count number of unique elements
        #    to construct an accurate cumulative discrete distribution func \sum_{x <= y_{i + 1}} 1/k
        y, counts = np.unique(np.round(x_norm[:,i], decimals=6), return_counts=True)
        p = 0
        # Ignore 0 and because v_min= 0
        for j in range(1, len(counts)):
            # integral from y_{i - 1} to y_{i} of |x - \sum_{x <= y_{i}} 1/k| dx
            yi1 = y[j - 1]
            yi = y[j]
            # Make a grid from yi1 to yi
            grid = np.linspace(yi1, yi, num=1000, endpoint=True)
            # Evaluate the integrand  |x - \sum_{x <= y_{i + 1}} 1/k|
            p += counts[j-1]
            integrand = np.abs(grid - p / num_mols)
            # Integrate using np.trapz
            wdu += np.trapz(y=integrand, x=grid)
        ans.append(wdu)
    return np.average(ans)


def hypersphere_overlap_of_subset(lib: np.ndarray, x: np.array) -> float:
    r"""Computes the overlap of subset with hyper-spheres around each point

    The edge penalty is also included, which disregards areas
    outside of the boundary of the full feature space/library.
    This is calculated as:

    .. math::
        g(S) = \sum_{i < j}^k O(i, j) + \sum^k_m E(m),

    where :math:`i, j` is over the subset of molecules,
    :math:`O(i, j)` is the approximate overlap between hyper-spheres,
    :math:`k` is the number of features and :math:`E`
    is the edge penalty of a molecule.

    Parameters
    ----------
    lib : ndarray
        Feature matrix of all molecules.
    x : ndarray
        Feature matrix of selected subset of molecules.

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
    # Find the maximum and minimum over each feature across all molecules.
    max_x = np.max(lib, axis=0)
    min_x = np.min(lib, axis=0)
    # Normalization of each feature to [0, 1]
    if np.any(np.abs(max_x - min_x) < 1e-30):
        raise ValueError(f"One of the features is redundant and causes normalization to fail.")
    x_norm = (x - min_x) / (max_x - min_x)
    # r_o = hypersphere radius
    r_o = d * np.sqrt(1 / k)
    if r_o > 0.5:
        warnings.warn(f"The number of molecules should be much larger"
                      " than the number of features.")
    g_s = 0
    edge = 0
    lam = (d - 1.0) / d   # Lambda parameter controls edge penalty
    # calculate overlap volume
    for i in range(0, (k - 1)):
        for j in range((i + 1), k):
            dist = np.linalg.norm(x_norm[i] - x_norm[j])
            # Overlap penalty
            if dist <= (2 * r_o):
                with np.errstate(divide='ignore'):
                    # min(100) ignores the inf case with divide by zero
                    g_s += min(100, 2 * (r_o / dist) - 1)
        # Edge penalty: lambda (1 - \sum^d_j e_{ij} / (dr_0)
        edge_pen = 0.0
        for j_dim in range(0, d):
            # calculate dist to closest boundary in jth coordinate,
            # with max value = 1, min value = 0
            dist_max = np.abs(1 - x_norm[i, j_dim])
            dist_min = x_norm[i, j_dim]
            dist = min(dist_min, dist_max)
            # truncate distance at r_o
            if dist > r_o:
                dist = r_o
            edge_pen += dist
        edge_pen /= (d * r_o)
        # print("Should be positive value only", (1.0 - edge_pen))
        edge_pen = lam * (1.0 - edge_pen)
        edge += edge_pen
    g_s += edge
    return g_s


def gini_coefficient(x: np.ndarray):
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
    x : ndarray(N, L)
        Molecule features in L bits with N molecules.

    Returns
    -------
    float :
        Gini coefficient between zero and one, where closer to zero indicates more diversity.

    References
    ----------
    Weidlich, Iwona E., and Igor V. Filippov. "Using the gini coefficient to measure the
    chemical diversity of small‐molecule libraries." (2016): 2091-2097.

    """
    # Check that `x` is a bit-wise fingerprint.
    if np.any(np.abs(np.sort(np.unique(x)) - np.array([0, 1])) > 1e-8):
        raise ValueError("Attribute `x` should have binary values.")
    if x.ndim != 2:
        raise ValueError(
            f"Attribute `x` should have dimension two rather than {x.ndim}."
        )

    numb_features = x.shape[1]
    # Take the bit-count of each column/molecule.
    bit_count = np.sum(x, axis=0)

    # Sort the bit-count since Gini coefficients relies on cumulative distribution.
    bit_count = np.sort(bit_count)

    # Mean of denominator
    denominator = numb_features * np.sum(bit_count)
    numerator = np.sum(np.arange(1, numb_features + 1) * bit_count)

    return 2.0 * numerator / denominator - (numb_features + 1) / numb_features
