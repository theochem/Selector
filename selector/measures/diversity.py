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

"""Molecule dataset diversity calculation module."""

import warnings

import numpy as np

from selector.measures.similarity import tanimoto

__all__ = [
    "compute_diversity",
    "logdet",
    "shannon_entropy",
    "explicit_diversity_index",
    "wdud",
    "hypersphere_overlap_of_subset",
    "gini_coefficient",
    "nearest_average_tanimoto",
]


def compute_diversity(
    feature_subset: np.array,
    div_type: str = "shannon_entropy",
    normalize: bool = False,
    truncation: bool = False,
    features: np.array = None,
    cs: int = None,
) -> float:
    """Compute diversity metrics.

    Parameters
    ----------
    feature_subset : np.ndarray
        Feature matrix.
    div_type : str, optional
        Method of calculation diversity for a given molecule set, which
        includes "entropy", "logdet", "shannon entropy", "wdud",
        "gini coefficient" "hypersphere_overlap", and
        "explicit diversity index".
        The default is "entropy".
    normalize : bool, optional
        Normalize the entropy to [0, 1]. Default is "False".
    truncation : bool, optional
        Use the truncated Shannon entropy. Default is "False".
    features : np.ndarray, optional
        Feature matrix of entire molecule library, used only if
        calculating `hypersphere_overlap_of_subset`. Default is "None".
    cs : int, optional
        Number of common substructures in molecular compound dataset.
        Used only if calculating `explicit_diversity_index`. Default is "None".


    Returns
    -------
    float, computed diversity.

    """
    func_dict = {
        "logdet": logdet,
        "wdud": wdud,
        "gini_coefficient": gini_coefficient,
    }

    if div_type in func_dict:
        return func_dict[div_type](feature_subset)

    # hypersphere overlap of subset
    elif div_type == "hypersphere_overlap":
        if features is None:
            raise ValueError(
                "Please input a feature matrix of the entire "
                "dataset when calculating hypersphere overlap."
            )
        return hypersphere_overlap_of_subset(features, feature_subset)

    elif div_type == "shannon_entropy":
        return shannon_entropy(feature_subset, normalize=normalize, truncation=truncation)

    elif div_type == "explicit_diversity_index":
        if cs is None:
            raise ValueError(
                "Attribute `cs` is missing. "
                "Please input `cs` value to use explicit_diversity_index."
            )
        elif cs == 0:
            raise ValueError("Divide by zero error: Attribute `cs` cannot be 0.")
        return explicit_diversity_index(feature_subset, cs)
    else:
        raise ValueError(f"Diversity type {div_type} not supported.")


def logdet(x: np.ndarray) -> float:
    r"""Compute the log determinant function.

    Given a  :math:`n_s \times n_f` feature matrix :math:`x`, where :math:`n_s` is the number of
    samples and :math:`n_f` is the number of features, the log determinant function is defined as:

    .. math:
        F_\text{logdet} = \log{\left(\det{\left(\mathbf{x}\mathbf{x}^T + \mathbf{I}\right)}\right)}

    where the :math:`I` is the :math:`n_s \times n_s` identity matrix.
    Higher values of :math:`F_\text{logdet}` mean more diversity.

    Parameters
    ----------
    x: ndarray of shape (n_samples, n_features)
        Feature matrix of `n_samples` samples in `n_features` dimensional feature space,

    Returns
    -------
    f_logdet: float
        The volume of parallelotope spanned by the matrix.

    Notes
    -----
    The log-determinant function is based on the formula in Nakamura, T., Sci Rep 2022.
    Please note that we used the
    natural logrithim to avoid the numerical stability issues,
    https://github.com/theochem/Selector/issues/229.

    References
    ----------
    Nakamura, T., Sakaue, S., Fujii, K., Harabuchi, Y., Maeda, S., and Iwata, S.., Selecting
    molecules with diverse structures and properties by maximizing submodular functions of
    descriptors learned with graph neural networks. Scientific Reports 12, 2022.

    """
    mid = np.dot(x, np.transpose(x)) + np.identity(x.shape[0])
    logdet_mid = np.linalg.slogdet(mid)
    f_logdet = logdet_mid.sign * logdet_mid.logabsdet
    return f_logdet


def shannon_entropy(x: np.ndarray, normalize=True, truncation=False) -> float:
    r"""Compute the shannon entropy of a binary matrix.

    Higher values mean more diversity.

    Parameters
    ----------
    x : ndarray
        Bit-string matrix.
    normalize : bool, optional
        Normalize the entropy to [0, 1]. Default=True.
    truncation : bool, optional
        Use the truncated Shannon entropy by only counting the contributions of one-bits.
        Default=False.

    Returns
    -------
    float :
        The shannon entropy of the matrix.

    Notes
    -----
    Suppose we have :math:`m` compounds and each compound has :math:`n` bits binary fingerprints.
    The binary matrix (feature matrix) is :math:`\mathbf{x} \in m \times n`, where each
    row is a compound and each column contains the :math:`n`-bit binary fingerprint.
    The equation for Shannon entropy is given by [1]_ and [3]_,

    .. math::
        H = \sum_i^m \left[ - p_i \log_2{p_i }  - (1 - p_i)\log_2(1 - p_i) \right]

    where :math:`p_i` represents the relative frequency of `1` bits at the fingerprint position
    :math:`i`. When :math:`p_i = 0` or :math:`p_i = 1`, the :math:`SE_i` is zero.
    When `completeness` is True, the entropy is calculated as in [2]_ instead

    .. math::
        H = \sum_i^m \left[ - p_i \log_2{p_i } \right]

    When `normalize` is True, the entropy is normalized by a scaling factor so that the entropy is in the range of
    [0, 1], [2]_

    .. math::
        H = \frac{ \sum_i^m \left[ - p_i \log_2{p_i }  - (1 - p_i)\log_2(1 - p_i) \right]}
            {n \log_2{2} / 2}

    But please note, when `completeness` is False and `normalize` is True, the formula has not been
    used in any literature. It is just a simple normalization of the entropy and the user can use it at their own risk.

    References
    ----------
    .. [1] Wang, Y., Geppert, H., & Bajorath, J. (2009). Shannon entropy-based fingerprint similarity
       search strategy. Journal of Chemical Information and Modeling, 49(7), 1687-1691.
    .. [2] Leguy, J., Glavatskikh, M., Cauchy, T., & Da Mota, B. (2021). Scalable estimator of the
       diversity for de novo molecular generation resulting in a more robust QM dataset (OD9) and a
       more efficient molecular optimization. Journal of Cheminformatics, 13(1), 1-17.
    .. [3] Weidlich, I. E., & Filippov, I. V. (2016). Using the Gini coefficient to measure the
       chemical diversity of small molecule libraries. Journal of Computational Chemistry, 37(22), 2091-2097.

    """
    # check if matrix is binary
    if np.count_nonzero((x != 0) & (x != 1)) != 0:
        raise ValueError("Attribute `x` should have binary values.")

    p_i_arr = np.sum(x, axis=0) / x.shape[0]
    h_x = 0

    for p_i in p_i_arr:
        if p_i == 0 or p_i == 1:
            # p_i = 0
            se_i = 0
        else:
            if truncation:
                # from https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00554-8
                se_i = -p_i * np.log2(p_i)
            else:
                # from https://pubs.acs.org/doi/10.1021/ci900159f
                se_i = -p_i * np.log2(p_i) - (1 - p_i) * np.log2(1 - p_i)

        h_x += se_i

    if normalize:
        if truncation:
            warnings.warn(
                "Computing the normalized Shannon entropy only counting the on-bits has not been reported in "
                "literature. The user can use it at their own risk."
            )

        h_x /= x.shape[1] * np.log2(2) / 2

    return h_x


# todo: add tests for edi
def explicit_diversity_index(
    x: np.ndarray,
    cs: int,
) -> float:
    """Compute the explicit diversity index.

    Parameters
    ----------
    x: ndarray of shape (n_samples, n_features)
        Feature matrix of `n_samples` samples in `n_features` dimensional feature space.
    cs : int
        Number of common substructures in the compound set.

    Returns
    -------
    edi_scaled : float
        Explicit diversity index.

    Notes
    -----
    This method hasn't been tested.

    This method is used only for datasets of molecular compounds.

    Papp, Á., Gulyás-Forró, A., Gulyás, Z., Dormán, G., Ürge, L.,
    and Darvas, F.. (2006) Explicit Diversity Index (EDI):
    A Novel Measure for Assessing the Diversity of Compound Databases.
    Journal of Chemical Information and Modeling 46, 1898-1904.
    """
    nc = len(x)
    sdi = (1 - nearest_average_tanimoto(x)) / (0.8047 - (0.065 * (np.log(nc))))
    cr = -1 * np.log10(nc / (cs**2))
    edi = (sdi + cr) * 0.7071067811865476
    edi_scaled = ((np.tanh(edi / 3) + 1) / 2) * 100
    return edi_scaled


def wdud(x: np.ndarray) -> float:
    r"""Compute the Wasserstein Distance to Uniform Distribution(WDUD).

    The equation for the Wasserstein Distance for a single feature to uniform distribution is

    .. math::
        WDUD(x) = \int_{0}^{1} |U(x) - V(x)|dx

    where the feature is normalized to [0, 1], :math:`U(x)=x` is the cumulative distribution
    of the uniform distribution on [0, 1], and :math:`V(x) = \sum_{y <= x}1 / N` is the discrete
    distribution of the values of the feature in :math:`x`, where :math:`y` is the ith feature. This
    integral is calculated iteratively between :math:`y_i` and :math:`y_{i+1}`, using trapezoidal method.

    Lower values of the WDUD mean more diversity because the features of the selected set are
    more evenly distributed over the range of feature values.

    Parameters
    ----------
    x: ndarray of shape (n_samples, n_features)
        Feature matrix of `n_samples` samples in `n_features` dimensional feature space.

    Returns
    -------
    float :
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

    # find the range of each feature
    col_diff = np.ptp(x, axis=0)
    # Normalization of each feature to [0, 1]
    if np.any(np.abs(col_diff) < 1e-30):
        # warning if some feature columns are constant
        warnings.warn(
            "Some of the features are constant which will cause the normalization to fail. "
            "Now removing them."
        )
        if np.all(col_diff < 1.0e-30):
            raise ValueError(
                "Unfortunately, all the features are constants and wdud cannot be calculated."
            )
        else:
            # remove the constant feature columns
            mask = np.ptp(x, axis=0) > 1e-30
            x = x[:, mask]
    x_norm = (x - np.min(x, axis=0)) / np.ptp(x, axis=0)

    # min_max normalization:
    n_samples, n_features = x_norm.shape
    ans = []  # store the Wasserstein distance for each feature
    for i in range(0, n_features):
        wdu = 0.0
        y = np.sort(x_norm[:, i])
        # Round to the sixth decimal place and count number of unique elements
        #    to construct an accurate cumulative discrete distribution func \sum_{x <= y_{i + 1}} 1/k
        y, counts = np.unique(np.round(x_norm[:, i], decimals=6), return_counts=True)
        p = 0
        # Ignore 0 and because v_min= 0
        for j in range(1, len(counts)):
            # integral from y_{i - 1} to y_{i} of |x - \sum_{x <= y_{i}} 1/k| dx
            yi1 = y[j - 1]
            yi = y[j]
            # Make a grid from yi1 to yi
            grid = np.linspace(yi1, yi, num=1000, endpoint=True)
            # Evaluate the integrand  |x - \sum_{x <= y_{i + 1}} 1/k|
            p += counts[j - 1]
            integrand = np.abs(grid - p / n_samples)
            # Integrate using np.trapz
            wdu += np.trapz(y=integrand, x=grid)
        ans.append(wdu)
    return np.average(ans)


def hypersphere_overlap_of_subset(x: np.ndarray, x_subset: np.array) -> float:
    r"""Compute the overlap of subset with hyper-spheres around each point

    The edge penalty is also included, which disregards areas
    outside of the boundary of the full feature space/library.
    This is calculated as:

    .. math::
        g(S) = \sum_{i < j}^k O(i, j) + \sum^k_m E(m)

    where :math:`i, j` is over the subset of molecules,
    :math:`O(i, j)` is the approximate overlap between hyperspheres,
    :math:`k` is the number of features and :math:`E`
    is the edge penalty of a molecule.

    Lower values mean more diversity.

    Parameters
    ----------
    x : ndarray
        Feature matrix of all molecules.
    x_subset : ndarray
        Feature matrix of selected subset of molecules.

    Returns
    -------
    float :
        The approximate overlapping volume of hyperspheres
        drawn around the selected points/molecules.

    Notes
    -----
    The hypersphere overlap volume is calculated using an approximation formula from Agrafiotis (1997).

    Agrafiotis, D. K.. (1997) Stochastic Algorithms for Maximizing Molecular Diversity.
    Journal of Chemical Information and Computer Sciences 37, 841-851.
    """

    # Find the maximum and minimum over each feature across all molecules.
    max_x = np.max(x, axis=0)
    min_x = np.min(x, axis=0)

    if np.all(np.abs(max_x - min_x) < 1e-30):
        raise ValueError("All of the features are redundant which causes normalization to fail.")

    # Remove redundant features
    non_red_feat = np.abs(max_x - min_x) > 1e-30
    x = x[:, non_red_feat]
    x_subset = x_subset[:, non_red_feat]
    max_x = max_x[non_red_feat]
    min_x = min_x[non_red_feat]

    d = len(x_subset[0])
    k = len(x_subset[:, 0])

    # normalization of each feature to [0, 1]
    x_norm = (x_subset - min_x) / (max_x - min_x)

    # r_o = hypersphere radius
    r_o = d * np.sqrt(1 / k)
    if r_o > 0.5:
        warnings.warn(
            "The number of molecules should be much larger" " than the number of features."
        )
    g_s = 0
    edge = 0

    # lambda parameter controls edge penalty
    lam = (d - 1.0) / d
    # calculate overlap volume
    for i in range(0, (k - 1)):
        for j in range((i + 1), k):
            dist = np.linalg.norm(x_norm[i] - x_norm[j])
            # Overlap penalty
            if dist <= (2 * r_o):
                with np.errstate(divide="ignore"):
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
        edge_pen /= d * r_o
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

    Lower values mean more diversity.

    Parameters
    ----------
    x : ndarray(N, L)
        Molecule features in L bits with N molecules.

    Returns
    -------
    float :
        Gini coefficient in the range [0,1].

    References
    ----------
    Weidlich, Iwona E., and Igor V. Filippov. "Using the gini coefficient to measure the
    chemical diversity of small‐molecule libraries." (2016): 2091-2097.

    """
    # Check that `x` is a bit-wise fingerprint.
    if np.count_nonzero((x != 0) & (x != 1)) != 0:
        raise ValueError("Attribute `x` should have binary values.")
    if x.ndim != 2:
        raise ValueError(f"Attribute `x` should have dimension two rather than {x.ndim}.")

    num_features = x.shape[1]
    # Take the bit-count of each column/molecule.
    bit_count = np.sum(x, axis=0)

    # Sort the bit-count since Gini coefficients relies on cumulative distribution.
    bit_count = np.sort(bit_count)

    # Mean of denominator
    denominator = num_features * np.sum(bit_count)
    numerator = np.sum(np.arange(1, num_features + 1) * bit_count)

    return 2.0 * numerator / denominator - (num_features + 1) / num_features


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
        # arbitrary distance for comparison:
        short = 100
        a = 0
        b = 0
        # search for shortest distance point from idx
        for jdx, _ in enumerate(x):
            dist = np.linalg.norm(x[idx] - x[jdx])
            if dist < short and idx != jdx:
                short = dist
                a = idx
                b = jdx
        # calculate tanimoto for each shortest dist pair
        tani.append(tanimoto(x[a], x[b]))
    # compute average of all shortest tanimoto coeffs
    nat = np.average(tani)
    return nat
