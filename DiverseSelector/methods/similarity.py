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

"""Module for Similarity-Based Selection Methods.

This module contains the classes and functions for the similarity-based selection methods. To select
a diverse subset of molecules the similarity-based selection methods select the molecules such that
the similarity between the molecules in the subset is minimized. The similarity of a set of
molecules is calculated using an n-array similarity index. These indexes compare n molecules at a
time and return a value between 0 and 1, where 0 means that all the molecules in the set are 
completely different and 1 means that the molecules are identical. 

The ideas behind the similarity-based selection methods are described in the following papers:
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00504-4

"""

from DiverseSelector.methods.base import SelectionBase
import numpy as np
import random
import math
from math import log
import warnings


class SimilarityIndex:
    r"""Calculate the n-ary similarity index of a set of vectors.

    This class provides methods for calculating the similarity index of a set of vectors represented
    as a matrix. Each vector is a row in the matrix, and each column represents a feature of the
    vector. The features in the vectors must be binary or real numbers between 0 and 1.

    Methods
    -------
    calculate_medoid(data, c_total=None, similarity_index=None, c_threshold=None, w_factor=None):
        Calculate the medoid of a set of real-valued vectors or binary objects. The similarity_index
        is used as the distance.

    calculate_outlier(data, c_total=None, similarity_index=None, c_threshold=None, w_factor=None):
        Calculate the outlier of a set of real-valued vectors or binary objects. The similarity_index
        is used as the distance.

    __call__(data=None, n_objects=None, similarity_index=None, c_threshold=None, w_factor=None):
        Calculate the similarity index of a set of vectors.

    """

    def __init__(self, similarity_index="RR", c_threshold=None, w_factor="fraction"):
        """Initialize the class.

        Parameters
        ----------
        similarity_index : str, optional
            The key with the abbreviation of the similarity index to be used for calculations.
            Possible values are:
                - 'AC': Austin-Colwell
                - 'BUB': Baroni-Urbani-Buser
                - 'CTn': Consoni-Todschini n
                - 'Fai': Faith
                - 'Gle': Gleason
                - 'Ja': Jaccard
                - 'Ja0': Jaccard 0-variant
                - 'JT': Jaccard-Tanimoto
                - 'RT': Rogers-Tanimoto
                - 'RR': Russel-Rao
                - 'SM': Sokal-Michener
                - 'SSn': Sokal-Sneath n
            Default is 'RR'.

        c_threshold : {None, 'dissimilar', int}, optional
            The coincidence threshold used for calculating similarity counters. A position in the elements
            is considered a coincidence (coincides among all the elements considered) if the number of
            elements that have the same value in that position is greater than the coincidence threshold.
                - None : Default, c_threshold = n_objects % 2
                - 'dissimilar' : c_threshold = ceil(n_objects / 2)
                - int : Integer number < n_objects

        w_factor : {"fraction", "power_n"}, optional
            The type of weight function to be used.
            - 'fraction' : similarity = d[k] / n, dissimilarity = 1 - (d[k] - n_objects % 2) / n_objects
            - 'power_n' : similarity = n ** -(n_objects - d[k]), dissimilarity = n ** -(d[k] - n_objects % 2)
            - other values : similarity = dissimilarity = 1
            Default is 'fraction'.
                other values : similarity = dissimilarity = 1
        """
        self.similarity_index = similarity_index
        self.w_factor = w_factor
        self.c_threshold = c_threshold
