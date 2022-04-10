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

from typing import Any

from DiverseSelector.utils import sklearn_supported_metrics
import numpy as np
import rdkit
from rdkit.Chem import rdFMCS
from scipy.spatial.distance import euclidean, squareform
from sklearn.metrics import pairwise_distances

__all__ = [
    "bit_tanimoto",
    "ComputeDistanceMatrix",
    "distance_to_similarity",
    "entropy",
    "euc_bit",
    "logdet",
    "modified_tanimoto",
    "pairwise_similarity_bit",
    "shannon_entropy",
    "tanimoto",
    "total_diversity_volume",
]


class ComputeDistanceMatrix:
    """Compute distance matrix.

    This class is just a demo and not finished yet.
    """

    def __init__(self,
                 feature: np.ndarray,
                 metric: str = "euclidean",
                 n_jobs: int = -1,
                 force_all_finite: bool = True,
                 **kwargs: Any,
                 ):
        """Compute pairwise distance given a feature matrix.

        Parameters
        ----------
        feature : np.ndarray
            Molecule feature matrix.
        metric : str, optional
            Distance metric.

        Returns
        -------
        dist : ndarray
            Symmetric distance array.
        """
        self.feature = feature
        self.metric = metric
        self.n_jobs = n_jobs
        self.force_all_finite = force_all_finite
        self.kwargs = kwargs

    def compute_distance(self):
        """Compute the distance matrix."""
        built_in_metrics = [
            "tanimoto",
            "modified_tanimoto",
        ]

        if self.metric in sklearn_supported_metrics:
            dist = pairwise_distances(
                X=self.feature,
                Y=None,
                metric=self.metric,
                n_jobs=self.n_jobs,
                force_all_finite=self.force_all_finite,
                **self.kwargs,
            )
        elif self.metric in built_in_metrics:
            func = self._select_function(self.metric)
            dist = func(self.feature)
        else:
            raise ValueError(f"Metric {self.metric} is not supported by the library.")

        return dist

    @staticmethod
    def _select_function(metric: str) -> Any:
        """Select the function to compute the distance matrix."""
        function_dict = {
            "tanimoto": tanimoto,
            "modified_tanimoto": modified_tanimoto,
        }

        return function_dict[metric]


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


def pairwise_similarity_bit(feature: np.array, metric: str) -> np.ndarray:
    """Compute the pairwaise similarity coefficients.

    Parameters
    ----------
    feature : ndarray
        Feature matrix.
    metric : str
        Method of calculation.

    Returns
    -------
    pair_coeff : ndarray
        Similairty coefficients for all molecule pairs in feature matrix.
    """
    pair_simi = []
    size = len(feature)
    for i in range(0, size):
        for j in range(i + 1, size):
            pair_simi.append(metric(feature[i], feature[j]))
    pair_coeff = (squareform(pair_simi) + np.identity(size))
    return pair_coeff


def euc_bit(a: np.array, b: np.array) -> float:
    """Compute Euclidean distance from bitstring.

    Parameters
    ----------
    a : array_like
        molecule A's features in bits.
    b : array_like
        molecules B's features in bits.

    Returns
    -------
    e_d : float
        Euclidean distance between molecule A and B.
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for idx, _ in enumerate(a):
        if a[idx] == b[idx] and a[idx] != 0:
            c += 1
    e_d = (a_feat + b_feat - (2 * c)) ** 0.5
    return e_d


def tanimoto(a: np.array, b: np.array) -> float:
    """Compute tanimoto coefficient.

    Parameters
    ----------
    a : array_like
        Molecule A's features.
    b : array_like
        Molecules B's features.

    Returns
    -------
    coeff : float
        Tanimoto coefficient for molecule A and B.
    """
    coeff = (sum(a * b)) / ((sum(a ** 2)) + (sum(b ** 2)) - (sum(a * b)))
    return coeff


def bit_tanimoto(a: np.array, b: np.array) -> float:
    """Compute tanimoto coefficient.

    Parameters
    ----------
    a : array_like
        Molecule A's features in bitstring.
    b : array_like
        Molecules B's features in bitstring.

    Returns
    -------
    coeff : float
        Tanimoto coefficient for molecule A and B.
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for idx, _ in enumerate(a):
        if a[idx] == b[idx] and a[idx] != 0:
            c += 1
    b_t = c / (a_feat + b_feat - c)
    return b_t


def modified_tanimoto(a: np.array, b: np.array) -> float:
    """Compute the modified tanimoto coefficient.

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
    """
    n = len(a)
    n_11 = sum(a * b)
    n_00 = sum((1 - a) * (1 - b))
    if n_00 == n:
        t_1 = 1
    else:
        t_1 = n_11 / (n - n_00)
    if n_11 == n:
        t_0 = 1
    else:
        t_0 = n_00 / (n - n_11)
    p = ((n - n_00) + n_11) / (2 * n)
    mt = (((2 - p) / 3) * t_1) + (((1 + p) / 3) * t_0)
    return mt


def entropy(x: np.ndarray) -> float:
    """Compute entropy of matrix.

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
    Feature matrixs are converted to bits,
    so we lose any information associated with num in matrix.
    """
    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            if x[i, j] != 0:
                x[i, j] = 1
            else:
                x[i, j] = 0
    length = len(x[0])
    n = len(x)
    top = 0
    val = []
    for i in range(0, length):
        val.append(sum(x[:, i]))
    ans = np.sort(val)
    for i in range(0, length):
        if ans[i] == 0:
            raise ValueError
        if ans[i] != 0:
            top += ((ans[i]) / n) * (np.log(ans[i] / n))
    e = -1 * (top / (length * 0.34657359027997264))
    return e


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
    This computes the tanimoto of pairs with the shortest distances,
    this is explictly for explicit diversity index.
    """
    tani = []
    for idx, _ in enumerate(x):
        short = 100  # arbitary distance
        a = 0
        b = 0
        for jdx, _ in enumerate(x):
            if euc_bit(x[idx], x[jdx]) < short and idx != jdx:
                short = euc_bit(x[idx], x[jdx])
                a = idx
                b = jdx
        tani.append(bit_tanimoto(x[a], x[b]))
    nat = np.average(tani)
    return nat


def explicit_diversity_index(x: np.ndarray, mol: rdkit.Chem.rdchem.Mol) -> float:
    """Computes the explicit diversity index.

    Parameters
    ----------
    x : ndarray
        Feature matrix.
    mol: rdkit.Chem.rdchem.Mol
        Molecules from feature matrix.

    Returns
    -------
    edi_scaled : float
        Explicit diversity index.
    """
    cs = len(rdFMCS.FindMCS(mol))
    nc = len(x)
    sdi = (1 - nearest_average_tanimoto(x)) / (0.8047 - (0.065 * (np.log(nc))))
    cr = -1 * np.log10(nc / (cs ** 2))
    edi = (sdi + cr) * 0.7071067811865476
    edi_scaled = ((np.tanh(edi / 3) + 1) / 2) * 100
    return edi_scaled


def logdet(x: np.ndarray) -> float:
    """Computes the log determinant function .

    Parameters
    ----------
    x : ndarray
        Subset feature matrix.

    Returns
    -------
    f_logdet: float
        The volume of parallelotope spand by the matrix.
    """
    mid = np.dot(np.transpose(x), x)
    f_logdet = np.log10(np.linalg.det(mid + np.identity(len(x[0]))))
    return f_logdet


def shannon_entropy(x: np.ndarray) -> float:
    """Computes the shannon entrop of a matrix.

    Parameters
    ----------
    x : ndarray
        Bit-string matrix.

    Returns
    -------
    h_x: float
        The shannon entropy of the matrix.
    """
    size = len(x[:, 0])
    h_x = 0
    for i in range(0, size):
        inter = np.count_nonzero(x[:, i]) / size
        if inter < (0.36787944117):
            h_x += (-1 * inter) * np.log10(inter)
        else:
            h_x += (-1 * inter) * np.log10(inter)
            # raise error
    return h_x


def wdud(x: np.ndarray) -> float:
    """Computes the Wasserstein Distance to Uniform Distribution(WDUD).

    Parameters
    ----------
    x : ndarray
        Feature matrix.

    Returns
    -------
    h_x: float
        The WDUD of the matrix.
    """
    # min_max normilization:
    d = len(x[0])
    n = len(x[:, 0])
    max_x = (max(map(max, x)))
    min_x = (min(map(min, x)))
    y = np.zeros((n, d))
    for i in range(0, len(x[:, 0])):
        for j in range(0, len(x[0])):
            y[i, j] = (x[i, j] - min_x) / (max_x - min_x)
    # wdud
    ans = []
    for i in range(0, d):
        h = -np.sort(-y[:, i])
        wdu = ((-1 / d) - h[0])
        for j in range(1, len(h)):
            wdu -= np.absolute(((j - 1) / d) - h[j])
        ans.append(wdu)
    return np.average(ans)


def total_diversity_volume(x: np.ndarray) -> float:
    """Computes the total diversity volume of the matrix.

    Parameters
    ----------
    x : ndarray
        Feature matrix.

    Returns
    -------
    g_s: float
        The total diversity volume of the matrix.
    """
    d = len(x[0])
    k = len(x[:, 0])
    # min_max normilization:
    max_x = (max(map(max, x)))
    min_x = (min(map(min, x)))
    y = np.zeros((k, d))
    for i in range(0, k):
        for j in range(0, d):
            y[i, j] = (x[i, j] - min_x) / (max_x - min_x)
    # divesity
    r_o = d * np.sqrt(1 / k)
    g_s = 0
    for i in range(0, (k - 1)):
        for j in range((i + 1), k):
            dist = euclidean(y[i], y[j])
            if dist <= (2 * r_o) and dist != 0:
                o_ij = min(100, 2 * r_o / dist - 1)
                g_s += o_ij
            else:
                o_ij = 0
                g_s += o_ij
    return g_s
