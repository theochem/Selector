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
#from rdkit import Chem
#from rdkit.Chem import MCS
from scipy.spatial.distance import cdist, squareform

sample2 = np.array([[1, 1, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])


__all__ = [
    "pairwise_dist",
    "compute_diversity",
    "distance_similarity",
    "pairwise_similarity",
    "pairwise_similarity_bit",
    "tanimoto",
    "cosine",
    "dice",
    "bit_tanimoto",
    "bit_cosine",
    "bit_dice",
]


def pairwise_dist(feature: np.array,
                  metric: str = "euclidean"):
    """Compute pairwise distance.

    Parameters
    ----------
    feature : ndarray
        feature matrix
    metric : str
        metric to use
    
    Returns
    -------
    arr_dist : ndarray
        symmetric distance array
    """
    if metric == "euclidean":
        arr_dist = cdist(feature, feature, "euclidean")
    else:
        arr_dist = cdist(feature, feature, metric)
    return arr_dist


def distance_similarity(distance: np.array):
    """Compute similarity.
    
    Parameters
    ----------
    distance : ndarray
        symmetric distance array

    Returns
    -------
    similarity : ndarray
        symmetric similarity array
    """
    similarity = 1 / (1 + distance)
    return similarity


def pairwise_similarity(feature: np.array, metric):
    """Compute the pairwaise similarity coefficients
    
    Parameters
    ----------
    feature : ndarray
        feature matrix
    metric : str
        method of calculation

    Returns
    -------
    pair_coeff : ndarray
        similairty coefficients for all molecule pairs in feature matrix
    """
    pair_simi = []
    size = len(np.shape(feature))
    for i in range(0, size + 1):
        for j in range(i + 1, size + 1):
            pair_simi.append((metric(feature[:,i], feature[:,j])))
    # this only works when the similarity to self is equal to 1
    pair_coeff = (squareform(pair_simi) + np.identity(size + 1))
    return pair_coeff


def pairwise_similarity_bit(feature: np.array, metric):
    """Compute the pairwaise similarity coefficients
    
    Parameters
    ----------
    feature : ndarray
        feature matrix in bit string
    metric : str
        method of calculation

    Returns
    -------
    pair_coeff : ndarray
        similairty coefficients for all molecule pairs in feature matrix
    """
    pair_simi = []
    size = len(feature)
    for i in range(0, size):
        for j in range(i + 1 , size):
            pair_simi.append(metric(feature[i],feature[j]))
    if metric == "euc_bit":
        pair_coeff = (squareform(pair_simi))
    else:
        pair_coeff = (squareform(pair_simi) + np.identity(size))
    return pair_coeff


def tanimoto(a, b):
    """Compute tanimoto coefficient

    Parameters
    ----------
    a : array_like
        molecule A's features
    b : array_like
        molecules B's features

    Returns
    -------
    coeff : int
        tanimoto coefficient for molecule A and B
    """
    coeff = (sum(a * b)) / ((sum(a ** 2)) + (sum(b ** 2)) - (sum(a * b)))
    return coeff 


def cosine(a, b):
    """Compute cosine coefficient

    Parameters
    ----------
    a : array_like
        molecule A's features
    b : array_like
        molecules B's features

    Returns
    -------
    coeff : int
        cosine coefficient for molecule A and B
    """
    coeff = (sum(a * b)) / (((sum(a ** 2)) + (sum(b ** 2))) ** 0.5)
    return coeff


def dice(a, b):
    """Compute dice coefficient

    Parameters
    ----------
    a : array_like
        molecule A's features
    b : array_like
        molecules B's features

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B
    """
    coeff = (2 * (sum(a * b))) / ((sum(a ** 2)) + (sum(b ** 2)))
    return coeff 


def modified_tanimoto(a, b):
    pass


def bit_tanimoto(a ,b):
    """Compute tanimoto coefficient

    Parameters
    ----------
    a : array_like
        molecule A's features in bitstring
    b : array_like
        molecules B's features in bitstring

    Returns
    -------
    coeff : int
        tanimoto coefficient for molecule A and B
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for i in range(0, len(a)):
        if a[i] == b[i] and a[i] != 0:
            c += 1
    b_t = c / (a_feat + b_feat - c)
    return b_t


def euc_bit(a, b):
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for i in range(0, len(a)):
        if a[i] == b[i] and a[i] != 0:
            c += 1
    e_d = (a_feat + b_feat - (2 * c)) ** 0.5
    return e_d


def bit_cosine(a ,b):
    """Compute dice coefficient

    Parameters
    ----------
    a : array_like
        molecule A's features in bit string
    b : array_like
        molecules B's features in bit string

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for i in range(0, len(a)):
        if a[i] == b[i] and a[i] != 0:
            c += 1
    b_c = c / ((a_feat * b_feat) ** 0.5)
    return b_c


def bit_dice(a ,b):
    """Compute dice coefficient

    Parameters
    ----------
    a : array_like
        molecule A's features
    b : array_like
        molecules B's features

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for i in range(0, len(a)):
        if a[i] == b[i] and a[i] != 0:
            c += 1
    b_d = (2 * c) / (a_feat + b_feat) 
    return b_d


def compute_diversity():
    """Compute the diversity."""
    pass


def total_diversity_volume():
    """Compute the total diversity volume."""
    pass


def gini(x):
    #returns negative values
    l = len(x[:,0])
    frac_top = 0
    frac_bottom = 0
    for i in range(0, l):
        frac_top += ((i + 1) * sum(x[i]))
        frac_bottom += (sum(x[i]))
    g = ((2 * frac_top) / (l * frac_bottom))- ((l + 1) / l)
    return g


def entropy(x):
    #returns negative values sometimes
    l = len(x[:,0])
    n = len(x)
    top = 0
    for i in range(0, l - 1):
        top += (sum(x[:,i]) / n) * (np.log(sum(x[:,i]) / n))
    e = -1 * top / (l * 0.34657359027997264)
    return e


def EDI(x):
    cs = len(MCS.FindMCS(x)) #place holder, input must be moleculues not matrix?
    nc = len(x)
    sdi = (1 - NAT(x)) / ( 0.8047 - (0.065 * (np.log(nc))))
    cr = -1 * np.log10(nc / (cs ** 2))
    edi = (sdi + cr) * 0.7071067811865476
    edi_scaled = ((np.tanh(edi / 3) + 1) / 2) * 100
    return edi_scaled


def NAT(x):
    tani = []
    for i in range(0, len(x)):
        short = 100 
        a = 0
        b = 0
        for j in range(0, len(x)):
            if euc_bit(x[i], x[j]) < short and i != j:
                short = euc_bit(x[i], x[j])
                a = i
                b = j
        tani.append(bit_tanimoto(x[a],x[b]))
    return np.average(tani)


def logdet(x):
    mid = np.dot(np.transpose(x) , x )
    f_logdet = np.linalg.det(mid + np.identity(len(x[0])))
    return f_logdet

