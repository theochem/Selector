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
import rdkit
from rdkit import Chem
from rdkit.Chem import MCS
from scipy.spatial.distance import squareform, cdist
from scipy import integrate

__all__ = [
    "pairwise_dist",
    "compute_diversity",
]


def pairwise_dist(feature: np.array,
                  metric: str = "euclidean"):
    """Compute pairwise distance."""
    # more to be implemented
    # https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/
    # scipy.spatial.distance.pdist.html?highlight=pdist#scipy.spatial.distance.pdist
    if metric == "euclidean":
        arr_dist = cdist(feature, feature, "euclidean")

    return arr_dist


def compute_diversity():
    """Compute the diversity."""
    pass


def total_diversity_volume():
    """Compute the total diversity volume."""

    pass


def gini(x):
    #numbers dont make sense
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
    #feature matrixs are conversted to bits
    #so we lose any information associated with num in matrix
    #if one of the features is zero for all molecules
    #it returns nan
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

def EDI(x, mol: rdkit.Chem.rdchem.Mol):
    CS = len(MCS.FindMCS(mol))
    NC = len(x)
    SDI = (1 - NAT(x)) / ( 0.8047 - (0.065 * (np.log(NC))))
    CR = -1 * np.log10(NC / (CS ** 2))
    edi = (SDI + CR) * 0.7071067811865476
    edi_scaled = ((np.tanh(edi / 3) + 1) / 2) * 100
    return edi_scaled

def pairwise_similarity_bit(feature: np.array, metric):
    pair_simi = []
    size = len(feature)
    for i in range(0, size):
        for j in range(i + 1 , size):
            pair_simi.append(metric(feature[i],feature[j]))
    pair_coeff = (squareform(pair_simi))
    return pair_coeff

def euc_bit(a, b):
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for i in range(0, len(a)):
        if a[i] == b[i] and a[i] != 0:
            c += 1
    e_d = (a_feat + b_feat - (2 * c)) ** 0.5
    return e_d

def bit_tanimoto(a ,b):
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for i in range(0, len(a)):
        if a[i] == b[i] and a[i] != 0:
            c += 1
    b_t = c / (a_feat + b_feat - c)
    return b_t

def NAT(x):
    tani = []
    for i in range(0, len(x)):
        short = 100 #arbitary large distance to beat
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
    #more volume means more diversity
    mid = np.dot(np.transpose(x) , x )
    f_logdet = np.linalg.det(mid + np.identity(len(x[0])))
    return f_logdet

def shannon_entropy(x):
    pass

def WDUD(x):
    #A smaller WDUD value implies that selected molecules 
    #are more diverse since the distribution of their 
    #property values is closer to being uniform
    pass
