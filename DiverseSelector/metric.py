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
from scipy.spatial.distance import cdist, pdist

__all__ = [
    "pairwise_dist",
    "compute_diversity",
]


def pairwise_dist(feature: np.array,
                  metric: str = "euclidean"):
    """Compute pairwise distance."""
    # more to be implemented
    # https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/referenc
    # e/generated/scipy.spatial.distance.pdist.html?highlight=pdist#scipy.spatial.distance.pdist
    if metric == "euclidean":
        arr_dist = cdist(feature, feature, "euclidean")
        # per_dist = pdist(feature ,"euclidean")
    return arr_dist


def euclidean(feature: np.array,
              a_feature: np.array,
              b_feature: np.array,
              metric: str = "euclidean",):
    """
    Compute the euclidean distance
    ----------
    Parameters
    ----------
    a_feature : Features in molecule A 
    b_feature : Features in molecule B
    feature : number of features shared by both
    -------
    Returns
    -------
    distance : euclidean distance between molecule A & B
    """
    if metric == "euclidean":
        if (a_feature) + (b_feature) - ((feature) * 2) > 0:
            distance = np.sqrt((a_feature) + (b_feature) - ((feature) * 2))
        else:
            raise ValueError("Error in input")
    return distance

def soergel(feature: np.array,
              a_feature: np.array,
              b_feature: np.array,
              metric: str = "soergel",):
    """
    Compute the soergel distance
    ----------
    Parameters
    ----------
    a_feature : Features in molecule A
    b_feature : Features in molecule B
    feature : number of features shared by both
    -------
    Returns
    -------
    distance : soergel distance between molecule A & B
    """
    if metric == "soergel":
        if (a_feature + b_feature - feature) != 0:
            distance = (a_feature + b_feature - (2 * feature)) / (a_feature + b_feature - feature)
            return distance
        else:
            raise ValueError("Error in input")


def compute_diversity(feature: np.array,
                      a_feature: np.array,
                      b_feature: np.array,
                      metric: str = "tanimoto"):
    """
    Compute the diversity.
    ----------
    Parameters
    ----------
    a_feature : Features in molecule A
    b_feature : Features in molecule B
    feature : number of features shared by both
    -------
    Returns
    -------
    diversity : diversity between molecule A & B
                1 being very diverse and 0 being not diverse
    """
    # diversity = 1 - similarity
    # similarity = 1 / 1 + distance
    if metric == "tanimoto":
        if ((a_feature) + (b_feature) - (feature)) != 0:
            similarity = (feature) /  (a_feature) + (b_feature) - (feature)
            diversity = 1 - similarity
            return diversity
        else:
            raise ValueError("Error in input")


def total_diversity_volume():
    """Compute the total diversity volume."""
    # V_tot = k * V_o
    pass
