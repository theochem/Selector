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
import numpy as np
import rdkit
from rdkit.Chem import MCS
from scipy.spatial.distance import cdist, squareform, euclidean
from sklearn.metrics import pairwise_distances
from DiverseSelector.utils import sklearn_supported_metrics
from DiverseSelector.test.common import euc_bit


__all__ = [
    "pairwise_dist",
    "distance_to_similarity",
    "pairwise_similarity_bit",
    "tanimoto",
    "bit_tanimoto",
]


class ComputeDistanceMatrix:
    """Compute distance matrix.

    This class is just a demo and not finished yet."""

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
            symmetric distance array.
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
        print(dist)
        return dist

    @staticmethod
    def _select_function(metric: str) -> Any:
        """Select the function to compute the distance matrix."""
        function_dict = {
            "tanimoto": tanimoto,
            "modified_tanimoto": modified_tanimoto,
        }

        return function_dict[metric]


def pairwise_dist(feature: np.array,
                  metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distance.

    Parameters
    ----------
    feature : ndarray
        feature matrix.
    metric : str
        method of calcualtion.

    Returns
    -------
    arr_dist : ndarray
        symmetric distance array.
    """
    return cdist(feature, feature, metric)


def distance_to_similarity(x, dist: bool = True) -> np.ndarray:
    """Convert between distance and similarity matrix.

    Parameters
    ----------
    distance : ndarray
        symmetric distance array.

    Returns
    -------
    similarity : ndarray
        symmetric similarity array.
    """
    if dist is True:
        y = 1 / (1 + x)
    else:
        y = (1 / x) - 1
    return y


def pairwise_similarity_bit(feature: np.array, metric) -> np.ndarray:
    """Compute the pairwaise similarity coefficients.

    Parameters
    ----------
    feature : ndarray
        feature matrix.
    metric : str
        method of calculation.

    Returns
    -------
    pair_coeff : ndarray
        similairty coefficients for all molecule pairs in feature matrix.
    """
    pair_simi = []
    size = len(feature)
    for i in range(0, size):
        for j in range(i + 1 , size):
            pair_simi.append(metric(feature[i],feature[j]))
    pair_coeff = (squareform(pair_simi) + np.identity(size))
    return pair_coeff


def tanimoto(a, b) -> int:
    """Compute tanimoto coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features.
    b : array_like
        molecules B's features.

    Returns
    -------
    coeff : int
        tanimoto coefficient for molecule A and B.
    """
    coeff = (sum(a * b)) / ((sum(a ** 2)) + (sum(b ** 2)) - (sum(a * b)))
    return coeff


def bit_tanimoto(a ,b) -> int:
    """Compute tanimoto coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features in bitstring.
    b : array_like
        molecules B's features in bitstring.

    Returns
    -------
    coeff : int
        tanimoto coefficient for molecule A and B.
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for idx, _ in enumerate(a):
        if a[idx] == b[idx] and a[idx] != 0:
            c += 1
    b_t = c / (a_feat + b_feat - c)
    return b_t


def modified_tanimoto(a, b) -> int:
    # incomplete
    n = len(a)
    n_11 = sum(a * b)
    n_00 = sum((1 - a) * (1 - b))
    if n_00 == n:
        t_1 = 1
    else:
        t_1 = n_11 / (n_00 - n)
    if n_11 == n:
        t_0 = 1
    else:
        t_0 = n_00 / (n - n_11)
    p = ((n - n_00) + n_11) / (2 * n)
    mt = (((2 - p) / 3) * t_1) + (((1 + p) / 3) * t_0)
    return mt


def gini(x):
    # incomplet
    # could be a wrong formula
    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            if x[i,j] != 0:
                x[i,j] = 1
            else:
                x[i,j] = 0
    L = len(x[0])
    frac_top = 0
    frac_bottom = 0
    val = []
    for i in range(0, L):
        val.append(sum(x[:,i]))
    ans = np.sort(val)
    for i in range(0, L):
        frac_top += ((i + 1) * ans[i])
        frac_bottom += ans[i]
    G = (2 * ((frac_top) / (L * frac_bottom))) - ((L + 1) / L)
    return G


def entropy(x):
    # note: feature matrixs are conversted to bits,
    # so we lose any information associated with num in matrix.
    for i in range(0, len(x)):
        for j in range(0, len(x[0])):
            if x[i,j] != 0:
                x[i,j] = 1
            else:
                x[i,j] = 0
    L = len(x[0])
    N = len(x)
    top = 0
    val = []
    for i in range(0, L):
        val.append(sum(x[:,i]))
    ans = np.sort(val)
    for i in range(0, L):
        if ans[i] == 0:
            raise ValueError
        else:
            top += ((ans[i]) / N ) * (np.log(ans[i] / N))
    E = -1 * (top / (L * 0.34657359027997264))
    return E


def nearest_average_tanimoto(x):
    tani = []
    for i in range(0, len(x)):
        short = 100 # arbitary distance
        a = 0
        b = 0
        for j in range(0, len(x)):
            if euc_bit(x[i], x[j]) < short and i != j:
                short = euc_bit(x[i], x[j])
                a = i
                b = j
        tani.append(bit_tanimoto(x[a],x[b]))
    return np.average(tani)


def explicit_diversity_index(x, mol: rdkit.Chem.rdchem.Mol):
    CS = len(MCS.FindMCS(mol))
    NC = len(x)
    SDI = (1 - nearest_average_tanimoto(x)) / ( 0.8047 - (0.065 * (np.log(NC))))
    CR = -1 * np.log10(NC / (CS ** 2))
    edi = (SDI + CR) * 0.7071067811865476
    edi_scaled = ((np.tanh(edi / 3) + 1) / 2) * 100
    return edi_scaled


def logdet(x):
    mid = np.dot(np.transpose(x) , x )
    f_logdet = np.linalg.det(mid + np.identity(len(x[0])))
    return f_logdet


def shannon_entropy(x):
    size = len(x[:,0])
    H_x = 0
    for i in range(0, size):
        inter =  np.count_nonzero(x[:,i]) / size
        if inter < (0.36787944117):
            H_x += (-1 * inter ) * np.log10(inter)
        else:
            raise ValueError
    return H_x


def wdud(x):
    # min_max normilization:
    d = len(x[0])
    n = len(x[:,0])
    max_x = (max(map(max, x)))
    y = np.zeros((n, d))
    for i in range(0, len(x[:,0])):
        for j in range(0, len(x[0])):
            y[i,j] = x[i,j] / max_x
    #wdud
    ans = []
    for i in range(0, d):
        h = -np.sort(-y[:,i])
        wdu = ((-1 / d) - h[0])
        for j in range(1, len(h)):
            wdu -= np.absolute(((j - 1) / d) - h[j])
        ans.append(wdu)
    return np.average(ans)


def total_diversity_Volume(x):
    d = len(x[0])
    k = len(x[:,0])
    # min_max normilization:
    max_x = (max(map(max, x)))
    y = np.zeros((k, d))
    for i in range(0, len(x[:,0])):
        for j in range(0, len(x[0])):
            y[i,j] = x[i,j] / max_x
    # divesity
    r_o = d * np.sqrt(1 / k)
    g_s = 0
    for i in range(0 , (k - 1)):
        for j in range((i + 1), k):
            dist = euclidean(y[i], y[j])
            if dist <= (2 * r_o) and dist != 0:
                O_ij = min(100, (((2 * r_o) / dist) - 1))
                g_s += O_ij
            else:
                pass
    return g_s
