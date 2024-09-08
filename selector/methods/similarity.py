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
"""Module for Similarity-Based Selection Methods.

This module contains the classes and functions for the similarity-based selection methods. To select
a diverse subset of molecules the similarity-based selection methods select the molecules such that
the similarity between the molecules in the subset is minimized. The similarity of a set of
molecules is calculated using an n-array similarity index. These indexes compare n molecules at a
time and return a value between 0 and 1, where 0 means that all the molecules in the set are
completely different and 1 means that the molecules are identical.

The ideas behind the similarity-based selection methods are described in the following papers:
    (esim)
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00504-4
    (isim)
    TODO: Add paper

"""

import math
import random
from math import log
from typing import List, Optional, Union
import warnings

import numpy as np

from selector.methods.base import SelectionBase

__all__ = [
    "NSimilarity",
    "SimilarityIndex",
]


class NSimilarity(SelectionBase):
    r"""Select samples of vectors using n-ary similarity indexes between vectors.

    The algorithms in this class select a diverse subset of vectors such that the similarity
    between the vectors in the subset is minimized. The similarity of a set of vectors is
    calculated using an n-ary similarity index. These indexes compare n vectors (e.g. molecular
    fingerprints) at a time and return a value between 0 and 1, where 0 means that all the vectors
    in the set are completely different and 1 means that the vectors are identical.

    The algorithm starts by selecting a starting reference data point. Then, the next data point is
    selected such as the similarity value of the group of selected data points is minimized. The
    process is repeated until the desired number of data points is selected.

    Notes
    -----
    The ideas behind the similarity-based selection methods are described in the following papers:

    """

    def __init__(
        self,
        method: str = "isim",
        inv_order: int = 1,
        similarity_index: str = "RR",
        w_factor: str = "fraction",
        c_threshold: Union[None, str, int] = None,
        preprocess_data: bool = True,
    ):
        """Initialize class.

        Parameters
        ----------
        method: {"isim", "esim"}
            Method used for calculating the similarity indices. The methods are:
                - "isim": Instant Similarity
                    The instant similarity index calculates the average similarity of a set of
                    objects without the need to calculate the similarity of all the possible pairs.
                    This method is easier to use than the ``esim`` (extended similarity index) as it
                    does not require the use of weight factors or coincidence thresholds.
                - "esim": Extended Similarity
                    The extended similarity index calculates the similarity of a set of objects
                    without the need to calculate the similarity of all the possible pairs. This
                    method requires the use of weight factors and coincidence thresholds.
        inv_order: int
            Integer indicating the 1/inv_order power used to approximate the average of the
            similarity values elevated to 1/inv_order.
        similarity_index: str
            Key with the abbreviation of the similarity index that will be used to perform the
            selection.
            Possible values are:
                AC: Austin-Colwell
                BUB: Baroni-Urbani-Buser
                CTn: Consoni-Todschini
                Fai: Faith
                Gle: Gleason
                Ja: Jaccard
                Ja0: Jaccard 0-variant
                JT: Jaccard-Tanimoto
                RT: Rogers-Tanimoto
                RR: Russel-Rao
                SM: Sokal-Michener
                SSn: Sokal-Sneath n
        w_factor: {"fraction", "power_n"}
            Type of weight function that will be used for calculating the counters.
                'fraction' : similarity = d[k]/n
                            dissimilarity = 1 - (d[k] - n_objects % 2)/n_objects
                'power_n' : similarity = n**-(n_objects - d[k])
                            dissimilarity = n**-(d[k] - n_objects % 2)
                other values : similarity = dissimilarity = 1
            Only used for the 'esim' method. Default is 'fraction'.
        c_threshold: {None, 'dissimilar', int}
            Coincidence threshold used for calculating the similarity counters. A column of the
            elements is considered to be a coincidence among the elements if the number of elements
            that have the same value in that position is greater than the coincidence threshold.
                None : Default, c_threshold = n_objects % 2
                'dissimilar' : c_threshold = ceil(n_objects / 2)
                int : Integer number < n_objects
            Only used for the 'esim' method. Default is None.
        preprocess_data: bool
            Every data element must be between 0 and 1 for the similarity indexes to work. If
            preprocess_data is True, the data is scaled between 0 and 1 using a strategy that is
            compatible with the similarity indexes. If preprocess_data is False, the data is not
            scaled and it is assumed that the data is already between 0 and 1.

        """
        # check if the method is valid
        if method not in ["isim", "esim"]:
            raise ValueError(f'Method "{method}" is not available please select "isim" or "esim".')
        # check if the similarity index is valid
        if similarity_index not in _similarity_index_dict:
            raise ValueError(
                f'Similarity index "{similarity_index}" is not available. '
                f"See the documentation for the available similarity indexes."
            )
        # for the esim method, check if the w_factor is valid
        if method == "esim":
            # check if the w_factor and c_threshold are valid
            if w_factor != "fraction":
                if w_factor.split("_")[0] != "power" or not w_factor.split("_")[-1].isdigit():
                    print(
                        f'Invalid weight factor "{w_factor}" given. Using default value '
                        '"similarity = dissimilarity = 1".'
                    )
                    w_factor = False
            # check if the c_threshold is valid
            if c_threshold not in ["dissimilar", None]:
                if not isinstance(c_threshold, int):
                    raise ValueError(
                        f'Invalid c_threshold. It must be an integer or "dissimilar" or None. '
                        f"Given c_threshold = {c_threshold}"
                    )

        self.method = method
        self.inv_order = inv_order
        self.similarity_index = similarity_index
        self.w_factor = w_factor
        self.c_threshold = c_threshold
        self.preprocess_data = preprocess_data
        # create an instance of the SimilarityIndex class. It is used to calculate the similarity
        # index of the sets of selected objects.
        self.si = SimilarityIndex(
            method=self.method,
            inv_order=self.inv_order,
            sim_index=self.similarity_index,
            c_threshold=self.c_threshold,
            w_factor=self.w_factor,
        )

    def _scale_data(self, X: np.ndarray):
        r"""Scales the data between so it can be used with the similarity indexes.

        First each data point is normalized to be between 0 and 1.

        .. math::
            x_{ij} = \frac{x_{ij} - min(x_j)}{max(x_j) - min(x_j)}

        Then, the average of each column is calculated. Finally, each element of the final working
        array will be defined as

        .. math::
            w_{ij} = 1 - | x_{ij} - a_j |

        where :math:`x_{ij}` is the element of the normalized array,
        and :math:`a_j` is the average of the j-th
        column of the normalized array.

        Parameters
        ----------
        X: np.ndarray
            Array of features (columns) for each sample (rows).
        """
        min_value = np.min(X)
        max_value = np.max(X)
        # normalize the data to be between 0 and 1 for working with the similarity indexes
        normalized_data = (X - min_value) / (max_value - min_value)
        # calculate the average of the columns
        col_average = np.average(normalized_data, axis=0)

        # each element of the final working array will be defined as w_ij = 1 - | x_ij - a_j |
        # where x_ij is the element of the normalized array, and a_j is the average of the j-th
        # column of the normalized array.
        data = 1 - np.abs(normalized_data - col_average)
        return data

    def _get_new_index(
        self,
        X: np.ndarray,
        selected_condensed: np.ndarray,
        num_selected: int,
        select_from: np.ndarray,
    ) -> int:
        r"""Select a new diverse sample from the data.

        The function selects a new sample such that the similarity of the new set of selected
        samples is minimized.

        Parameters
        ----------
        X: np.ndarray
            Array of features (columns) for each sample (rows).
        selected_condensed: np.ndarray
            Columnwise sum of all the samples selected so far.
        num_selected: int
            Number of samples selected so far.
        select_from: np.ndarray
            Array of integers representing the indices of the samples that have not been selected
            yet.

        Returns
        -------
        selected: int
            Index of the new selected sample.

        """
        # check if the data was previously scaled
        if np.max(X) > 1 or np.min(X) < 0:
            raise ValueError(
                "The data was not scaled between 0 and 1. "
                "Use the _scale_data function to scale the data."
            )

        # Number of total vectors used to calculate th similarity. It is the number of samples
        # selected so far + 1, because the similarities are computed for the sets of samples after
        # a new selection is made.
        n_total = num_selected + 1

        # min value that is guaranteed to be higher than all the comparisons, this value should be a
        # warranty that a exist a set of samples with similarity lower than min_value. The max
        # possible similarity value for set of samples is 1.00.
        min_value = 1.01

        # placeholder index, initiating variable with a number outside the possible index values
        index = X.shape[0] + 1

        # for all indices that have not been selected
        for sample_idx in select_from:
            # column sum
            c_total = selected_condensed + X[sample_idx]

            # calculating similarity
            sim_index = self.si(c_total, nsamples=n_total)

            # if the sim of the set is less than the similarity of the previous diverse set,
            # update min_value and index
            if sim_index < min_value:
                index = sample_idx
                min_value = sim_index

        return index

    def calculate_medoid(self, X: np.ndarray, c_total=None) -> int:
        """Calculate the medoid of a set of real-valued vectors or binary objects.

        Parameters
        ----------
        X: np.array
            np.array of all the real-valued vectors or binary objects.
        c_total:
            np.array with the columnwise sums of the data, not necessary to provide.
        """
        # Check if the data is a np.ndarray of a list
        if not isinstance(X, np.ndarray):
            raise TypeError("Input data is not a np.ndarray, please input the right data type")
        # Check if the data is one dimensional
        if X.ndim != 2:
            raise ValueError(
                "Data must be a two dimensional np.ndarray for calculating the medoid."
            )
        # Check if the data has at least 3 rows
        if X.shape[0] < 3:
            raise ValueError("Input data must have at least 3 rows to calculate the medoid.")

        # check if c_total is provided and if not, calculate it
        if c_total is None:
            c_total = np.sum(X, axis=0)
        # if c_total is provided, check if it has the same number of columns as the data
        elif c_total is not None and len(X[0]) != len(c_total):
            raise ValueError("Dimensions of objects and columnwise sum differ")

        # get the total number of objects
        nsamples = X.shape[0]

        # Initialize the selected index with a number outside the possible index values
        index = nsamples + 1

        # minimum similarity value that is guaranteed to be higher than all the comparisons, this
        # value should be a warranty that a exist a sample with similarity lower than min_sim. The
        # max possible similarity value for set of samples is 1.00.
        min_sim = 1.01

        # For each sample in the set, calculate the column-wise sum of the data without the sample
        comp_sums = c_total - X

        # The medoid is calculated using an instance of the SimilarityIndex class with the same
        # parameters as the current class, but with the inv_order = 1
        si = SimilarityIndex(
            method=self.method,
            inv_order=1,
            sim_index=self.similarity_index,
            c_threshold=self.c_threshold,
            w_factor=self.w_factor,
        )

        # for each sample calculate the similarity index of the complete set without the sample
        for idx, row in enumerate(comp_sums):
            # calculate the similarity index of the set of objects without the current object
            sim_index = si(row, nsamples=nsamples - 1)
            # if the similarity is lower than the previous minimum similarity, update the minimum
            # similarity and the index
            if sim_index < min_sim:
                min_sim, index = sim_index, idx
        # the index of the object that increases more the similarity of the set when added is
        # returned (the medoid)
        return index

    def calculate_outlier(self, X: np.ndarray = None, c_total=None) -> int:
        r"""Calculate the outlier of a set of real-valued vectors or binary objects.

        Calculates the outlier of a set of real-valued vectors or binary objects. Using the
        similarity index provided in the class initialization.

        Parameters
        ----------
        X: np.array
            np.array of all the real-valued vectors or binary objects.
        c_total: np.array, optional
            np.array with the column-wise sums of the data.
        """
        # Check if the data is a np.ndarray of a list
        if not isinstance(X, np.ndarray):
            raise TypeError("Input data is not a np.ndarray, please input the right data type")
        # Check if the data is one dimensional
        if X.ndim != 2:
            raise ValueError(
                "Data must be a two dimensional np.ndarray for calculating the outlier."
            )
        # Check if the data has at least 3 rows
        if X.shape[0] < 3:
            raise ValueError("Input data must have at least 3 rows to calculate the outlier.")

        # check if c_total is provided and if not, calculate it
        if c_total is None:
            c_total = np.sum(X, axis=0)
        # if c_total is provided, check if it has the same number of columns as the data
        elif c_total is not None and len(X[0]) != len(c_total):
            raise ValueError("Dimensions of objects and columnwise sum differ")

        nsamples = X.shape[0]

        # Initialize the selected index with a number outside the possible index values
        index = nsamples + 1

        # maximum similarity value that is guaranteed to be lower than all the comparisons, this
        # value should be a warranty that a exist a sample with similarity lower than min_sim. The
        # min possible similarity value for set of samples is 0.00.
        max_sim = -0.01

        # For each sample in the set, calculate the columnwise sum of the data without the sample
        comp_sums = c_total - X

        # The outlier is calculated using an instance of the SimilarityIndex class with the same
        # parameters as the current class, but with the inv_order = 1

        # for each sample calculate the similarity index of the complete set without the sample
        for idx, obj in enumerate(comp_sums):
            # calculate the similarity index of the set of objects without the current object
            sim_index = self.si(X=obj, nsamples=nsamples - 1)
            # if the similarity is bigger than the previous minimum similarity, update the minimum
            # similarity and the index
            if sim_index > max_sim:
                max_sim, index = sim_index, idx
        # the index of the object that decreases more the similarity of the set when added is
        # returned (the outlier)
        return index

    def select_from_cluster(
        self,
        X: np.ndarray,
        size: int,
        labels: Optional[np.ndarray] = None,
        start: Union[str, List[int]] = "medoid",
    ) -> List[int]:
        r"""Algorithm of nary similarity selection for selecting points from cluster.

        Parameters
        ----------
        X: np.ndarray
            Array of features (columns) for each sample (rows).
        size: int
            Number of sample points to select (i.e. size of the subset).
        labels: np.ndarray, optional
            Array of integers or strings representing the points ids of the data that belong to the
            current cluster. If `None`, all the samples in the data are treated as one cluster.
        start: str or list
            srt: key on what is used to start the selection {'medoid', 'random', 'outlier'}.
            list: indices of points that are included in the selection since the beginning.

        Returns
        -------
        selected: list
            Indices of the selected sample points.

        """
        # check for valid start value and raise an error if it is not
        if start not in ["medoid", "random", "outlier"]:
            if not isinstance(start, list) or not all(isinstance(i, int) for i in start):
                raise ValueError(
                    "Select a correct starting point: medoid, random, outlier or a list of indices."
                )
        # check if cluster_ids are provided
        if labels is not None:
            # extract the data corresponding to the cluster_ids
            X = np.take(X, labels, axis=0)

        # total number of objects in the current cluster
        nsamples = X.shape[0]
        # ids of the data points in the current cluster (0, 1, 2, ..., samples-1)
        data_ids = np.array(range(nsamples))

        # check if the number of selected objects is less than the total number of objects
        if size > nsamples:
            raise ValueError(
                f"Number of samples is less than the requested sample size: {nsamples} < {size}."
            )

        # The data is marked to be preprocessed scale the data between 0 and 1 using a strategy
        # that is compatible with the similarity indexes
        if self.preprocess_data:
            X = self._scale_data(X)
        else:
            # check if the data is between 0 and 1 and raise an error if it is not
            if np.max(X) > 1 or np.min(X) < 0:
                raise ValueError(
                    "The data was not scaled between 0 and 1. "
                    "Use the _scale_data function to scale the data."
                )

        # select the index (of the working data) corresponding to the medoid of the data using the
        # similarity index
        if start == "medoid":
            seed = self.calculate_medoid(X)
            selected = [seed]
        # select the index (of the working data)  corresponding to a random data point
        elif start == "random":
            seed = random.randint(0, nsamples - 1)
            selected = [seed]
        # select the index (of the working data) corresponding to the outlier of the data using the
        # similarity index
        elif start == "outlier":
            seed = self.calculate_outlier(X)
            selected = [seed]
        # if a list of cluster_ids is provided, select the data_ids corresponding indices
        elif isinstance(start, list):
            if labels is not None:
                # check if all starting indices are in this cluster
                if not all(label in labels for label in start):
                    raise ValueError(
                        "Some of the provided initial indexes are not in the cluster data."
                    )
                # select the indices of the data_ids that correspond to the provided starting points
                # provided from cluster_ids
                selected = [i for i, j in enumerate(labels) if j in start]
            else:
                # check if all starting indices are in the data
                if not all(label in data_ids for label in start):
                    raise ValueError("Some of the provided initial indexes are not in the data.")
                # select the indices of the data_ids that correspond to the provided starting points
                selected = start[:]
        # Number of initial objects
        num_selected = len(selected)

        # get selected samples form the working data array
        selected_objects = np.take(X, selected, axis=0)
        # Calculate the columnwise sum of the selected samples
        selected_condensed = np.sum(selected_objects, axis=0)

        # until the desired number of objects is selected a new object is selected.
        while num_selected < size:
            # indices from which to select the new data points
            select_from = np.delete(data_ids, selected)

            # Select new index. The new object is selected such that from all possible objects the
            # similarity of the set of (selected_objects + new_object) is a minimum.
            new_index = self._get_new_index(X, selected_condensed, num_selected, select_from)

            # updating column sum vector
            selected_condensed += X[new_index]

            # updating selected indices
            selected.append(new_index)
            num_selected += 1

        return selected


class SimilarityIndex:
    r"""Calculate the n-ary similarity index of a set of vectors.

    This class provides methods for calculating the similarity index of a set of vectors represented
    as a matrix. Each vector is a row in the matrix, and each column represents a feature of the
    vector. The features in the vectors must be binary or real numbers between 0 and 1.
    """

    def __init__(
        self,
        method: str = "isim",
        inv_order: int = 1,
        sim_index: str = "RR",
        c_threshold: Union[None, str, int] = None,
        w_factor: str = "fraction",
    ):
        """Initialize the class.

        Parameters
        ----------
        method : {"isim", "esim"}, optional
            Method used for calculating the similarity indexes. The methods are:
                - "isim": Instant Similarity
                    The instant similarity index calculates the average similarity of a set of
                    objects without the need to calculate the similarity of all the possible pairs.
                    This method is easier to use than the ``esim`` (extended similarity index) as it
                    does not require the use of weight factors or coincidence thresholds.
                - "esim": Extended Similarity
                    The extended similarity index calculates the similarity of a set of objects
                    without the need to calculate the similarity of all the possible pairs. This
                    method requires the use of weight factors and coincidence thresholds.
            Default is "isim".

        inv_order : int, optional
            Integer indicating the 1/inv_order power used to approximate the average of the
            similarity values elevated to 1/inv_order. This is not used for calculating the
            medoid or the outlier. Default is 1.

        sim_index : str, optional
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
            The coincidence threshold used for calculating similarity counters. A position in the
            elements is considered a coincidence (coincides among all the elements considered) if
            the number of elements that have the same value in that position is greater than the
            coincidence threshold.
                - None : Default, c_threshold = n_objects % 2
                - 'dissimilar' : c_threshold = ceil(n_objects / 2)
                - int : Integer number < n_objects
            It is only used for the 'esim' method.

        w_factor : {"fraction", "power_n"}, optional
            The type of weight function to be used.
            - 'fraction' :
                similarity = d[k] / n, dissimilarity = 1 - (d[k] - n_objects % 2) / n_objects
            - 'power_n' :
                similarity = n ** -(n_objects - d[k]), dissimilarity = n ** -(d[k] - n_objects % 2)
            - other values :
                similarity = dissimilarity = 1
            Default is 'fraction'.
                other values : similarity = dissimilarity = 1
            It is only used for the 'esim' method. Default is 'fraction'.
        """
        # check if the method is valid
        if method not in ["isim", "esim"]:
            raise ValueError(f'Method "{method}" is not available please select "isim" or "esim".')

        # check if the similarity index is valid
        if sim_index not in _similarity_index_dict:
            raise ValueError(
                f'Similarity index "{sim_index}" is not available. '
                f"See the documentation for the available similarity indexes."
            )
        # for the esim method, check if the w_factor and c_threshold are valid
        if method == "esim":
            # check if the c_threshold is valid
            if c_threshold not in ["dissimilar", None]:
                if not isinstance(c_threshold, int):
                    raise ValueError(
                        f'Invalid c_threshold. It must be an integer or "dissimilar" or None. '
                        f"Given c_threshold = {c_threshold}"
                    )
            # check if the w_factor is valid
            if w_factor != "fraction":
                if w_factor.split("_")[0] != "power" or not w_factor.split("_")[-1].isdigit():
                    print(
                        f'Invalid weight factor "{w_factor}" given. Using default value '
                        '"similarity = dissimilarity = 1".'
                    )
                    w_factor = False
        self.method = method
        self.inv_order = inv_order
        self.similarity_index = sim_index
        self.w_factor = w_factor
        self.c_threshold = c_threshold

    def _calculate_counters(self, X: np.ndarray, nsamples: Optional[int] = None) -> dict:
        """Calculate 1-similarity, 0-similarity, and dissimilarity counters.

        Arguments
        ---------
        X : np.ndarray
            Array of arrays, each sub-array contains the binary or real valued vector. The values
            must be between 0 and 1. If the number of rows ==1, the data is treated as the
            columnwise sum of the objects. If the number of rows > 1, the data is treated as the
            objects.
        nsamples: int
            Number of objects, only necessary if the columnwise sum of the objects is provided
            instead of the data (num rows== 1). If the data is provided, the number of objects is
            calculated as the length of the data.

        Returns
        -------
        counters : dict
            Dictionary with the weighted and non-weighted counters.
        """
        # Check if the data is a np.ndarray of a list
        if not isinstance(X, np.ndarray):
            raise TypeError("Argument X is not a np.ndarray, please input the right data type!")
        # Check if data is a columnwise sum or the objects
        if X.ndim == 1:
            c_total = X
            # If data is a columnwise sum, check if n_objects is provided
            if nsamples is None:
                raise ValueError("Argument X is the columnwise sum, please specify nsamples")
        else:
            c_total = np.sum(X, axis=0)
            if nsamples is not None and nsamples != len(X):
                warnings.warn(
                    f"Warning, specified number of objects {nsamples} is different from the number"
                    " of objects in data {len_arr}\n"
                    "Doing calculations with",
                    nsamples,
                    "objects.",
                )
            nsamples = len(X)

        # if method is isim, calculate the counters using the instant similarity index
        if self.method == "isim":
            # calculate number of instances with common on bits (common 1s) for each column
            a_array = c_total * (c_total - 1) / 2
            # calculate number of instances with common off bits (common 0s) for each column
            off_coincidences = nsamples - c_total
            d_array = off_coincidences * (off_coincidences - 1) / 2
            # calculate number of instances with dissimilar bits for each column
            dis_array = off_coincidences * c_total

            # calculate total a, d, b + c counters
            a = np.sum(np.power(a_array, 1.0 / self.inv_order))
            d = np.sum(np.power(d_array, 1.0 / self.inv_order))
            total_dis = np.sum(np.power(dis_array, 1 / self.inv_order))

            # calculate total similarity, and total similarity + dissimilarity
            total_sim = a + d
            p = total_sim + total_dis

            # in the isim method, the counters are not weighted, so the weighted counters are equal
            # to the non-weighted counters
            w_a = a
            w_d = d
            total_w_sim = total_sim
            total_w_dis = total_dis
            w_p = p

        elif self.method == "esim":
            # Assign c_threshold
            if self.c_threshold is None:
                tmp_c_threshold = nsamples % 2
            elif self.c_threshold == "dissimilar":
                tmp_c_threshold = math.ceil(nsamples / 2)
            elif isinstance(self.c_threshold, int):
                if self.c_threshold >= nsamples:
                    raise ValueError(
                        "c_threshold cannot be equal or greater than nsamples. \n"
                        f"c_threshold = {self.c_threshold}  nsamples = {nsamples}"
                    )
                tmp_c_threshold = self.c_threshold
            else:
                raise ValueError(
                    "c_threshold must be None, 'dissimilar' or an integer. \n"
                    f"Given c_threshold = {self.c_threshold}"
                )

            # Calculate a, d, b + c
            # Calculate the positions (columns) of common on bits (common 1s) between the objects
            a_indices = 2 * c_total - nsamples > tmp_c_threshold
            # Calculate the positions (columns) common off bits (common 0s) between the objects
            d_indices = nsamples - 2 * c_total > tmp_c_threshold
            # Calculate the positions (columns) of dissimilar bits between the objects (b + c)
            # the dissimilar bits are the bits that are not common between the objects
            dis_indices = np.abs(2 * c_total - nsamples) <= tmp_c_threshold

            # Calculate the number of columns with common on bits (common 1s) between the objects
            a = np.sum(np.power(a_indices, 1.0 / self.inv_order))
            # Calculate the number of columns with common off bits (common 0s) between the objects
            d = np.sum(np.power(d_indices, 1.0 / self.inv_order))
            # Calculate the number of columns with dissimilar bits between the objects (b + c)
            total_dis = np.sum(np.power(dis_indices, 1.0 / self.inv_order))

            # calculate the weights for each column indexed as with common on bits (common 1s)
            a_w_array = self._f_s(2 * c_total[a_indices] - nsamples, nsamples)
            # calculate the weights for each column indexed as with common off bits (common 0s)
            d_w_array = self._f_s(abs(2 * c_total[d_indices] - nsamples), nsamples)
            # calculate the weights for each column indexed as with dissimilar bits
            total_w_dis_array = self._f_d(abs(2 * c_total[dis_indices] - nsamples), nsamples)

            # calculate the total weight for each type of counter
            w_a = np.sum(np.power(a_w_array, 1.0 / self.inv_order))
            w_d = np.sum(np.power(d_w_array, 1.0 / self.inv_order))
            total_w_dis = np.sum(np.power(total_w_dis_array, 1.0 / self.inv_order))

            # calculate the counters needed to calculate the similarity indexes
            total_sim = a + d
            total_w_sim = w_a + w_d
            p = total_sim + total_dis
            w_p = total_w_sim + total_w_dis

        counters = {
            "a": a,
            "w_a": w_a,
            "d": d,
            "w_d": w_d,
            "total_sim": total_sim,
            "total_w_sim": total_w_sim,
            "total_dis": total_dis,
            "total_w_dis": total_w_dis,
            "p": p,
            "w_p": w_p,
        }
        return counters

    def _f_s(self, d, n) -> float:
        """Calculate the similarity weight factor for a given number of similar objects in a set.

        Parameters
        ----------
        d : int
            Number of similar objects.
        n : int
            Total number of objects.

        Returns
        -------
        w_s : float
            Weight factor for the similarity depending on the number of objects that are similar (d)
            in a set of (n) objects.
        """
        if self.w_factor:
            # power_n case
            if "power" in self.w_factor:
                power = int(self.w_factor.split("_")[-1])
                return power ** -(n - d).astype(float)
            # fraction case
            elif self.w_factor == "fraction":
                return d / n
            else:
                raise ValueError(
                    f"w_factor must be 'fraction' or 'power_n'. \n Given w_factor = {self.w_factor}"
                )
        # default case, the similarity counters are not weighted
        return 1

    def _f_d(self, d, n) -> float:
        """Calculate the dissimilarity weight factor for a given number of similar objects in a set.

        Parameters
        ----------
        d : int
            Number of similar objects.
        n : int
            Total number of objects.

        Returns
        -------
        w_s : float
            Weight factor for the dissimilarity depending on the number of objects that are similar
            (d) in a set of (n) objects.
        """
        if self.w_factor:
            # power_n case
            if "power" in self.w_factor:
                power = int(self.w_factor.split("_")[-1])
                return power ** -(d - n % 2).astype(float)
            # fraction case
            elif self.w_factor == "fraction":
                return 1 - (d - n % 2) / n
            else:
                raise ValueError(
                    f"w_factor must be 'fraction' or 'power_n'. \n Given w_factor = {self.w_factor}"
                )
        # default case, the dissimilarity counters are not weighted
        return 1

    def __call__(self, X: np.ndarray, nsamples: int = None) -> float:
        """Calculate the similarity index of a set of vectors.

        Parameters
        ----------
        X : np.ndarray
            Array of arrays, each sub-array contains the binary or real valued vector. The values
            must be between 0 and 1. If the number of rows ==1, the data is treated as the
            columnwise sum of the objects. If the number of rows > 1, the data is treated as the
            objects.
        nsamples: int
            Number of objects in the data. Is only necessary if the data is a columnwise sum of
            the objects. If the data is not the columnwise sum of the objects, the number of objects
            is calculated as the length of the data.
        Returns
        -------
        similarity_index: float
            Similarity index of the set of vectors.
        """
        # check if arr is a np.ndarray
        if not isinstance(X, np.ndarray):
            raise TypeError("Input data is not a np.ndarray, please input the right data type")

        # if the data is a columnwise sum of the objects check that n_objects is provided
        if X.ndim == 1:
            c_total = X
            if not nsamples:
                raise ValueError("Input data is the columnwise sum, please specify nsamples")
        # if the data is not a columnwise sum of the objects, calculate the columnwise sum and the
        # number of objects
        else:
            c_total = np.sum(X, axis=0)
            if nsamples is not None and nsamples != len(X):
                warnings.warn(
                    f"Warning, specified number of objects {nsamples} is different from the number"
                    " of objects in data {len_arr}\n"
                    "Doing calculations with",
                    nsamples,
                    "objects.",
                )
            nsamples = X.shape[0]

        # calculate the counters needed to calculate the similarity indexes
        counters = self._calculate_counters(X=c_total, nsamples=nsamples)
        # calculate the similarity index
        similarity_index = _similarity_index_dict[self.similarity_index](counters)

        return similarity_index


# Utility functions section
# -------------------------

# Functions that calculate the similarity indexes. The functions are named as the similarity
# index they calculate. The _nw suffix indicates that the similarity index is not weighted.
# More information about the similarity indexes can be found in the following paper:


def _ac_nw(counters: dict) -> float:
    """Calculate the Austin-Colwell (AC) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    ac_nw = (2 / np.pi) * np.arcsin(np.sqrt(counters["total_w_sim"] / counters["p"]))
    return ac_nw


def _bub_nw(counters: dict) -> float:
    """Calculate the Baroni-Urbani-Buser (BUB) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    bub_nw = ((counters["w_a"] * counters["w_d"]) ** 0.5 + counters["w_a"]) / (
        (counters["a"] * counters["d"]) ** 0.5 + counters["a"] + counters["total_dis"]
    )
    return bub_nw


def _ct1_nw(counters: dict) -> float:
    """Calculate the Consoni-Todschini 1 (CT1) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    ct1_nw = (log(1 + counters["w_a"] + counters["w_d"])) / (log(1 + counters["p"]))
    return ct1_nw


def _ct2_nw(counters: dict) -> float:
    """Calculate the Consoni-Todschini 2 (CT2) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    ct2_nw = (log(1 + counters["w_p"]) - log(1 + counters["total_w_dis"])) / (
        log(1 + counters["p"])
    )
    return ct2_nw


def _ct3_nw(counters: dict) -> float:
    """Calculate the Consoni-Todschini 3 (CT3) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    ct3_nw = (log(1 + counters["w_a"])) / (log(1 + counters["p"]))
    return ct3_nw


def _ct4_nw(counters: dict) -> float:
    """Calculate the Consoni-Todschini 4 (CT4) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    ct4_nw = (log(1 + counters["w_a"])) / (log(1 + counters["a"] + counters["total_dis"]))
    return ct4_nw


def _fai_nw(counters: dict) -> float:
    """Calculate the Faith (Fai) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    fai_nw = (counters["w_a"] + 0.5 * counters["w_d"]) / (counters["p"])
    return fai_nw


def _gle_nw(counters: dict) -> float:
    """Calculate the Gleason (Gle) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    gle_nw = (2 * counters["w_a"]) / (2 * counters["a"] + counters["total_dis"])
    return gle_nw


def _ja_nw(counters: dict) -> float:
    """Calculate the Jaccard (Ja) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    ja_nw = (3 * counters["w_a"]) / (3 * counters["a"] + counters["total_dis"])
    return ja_nw


def _ja0_nw(counters: dict) -> float:
    """Calculate the Jaccard 0-variant (Ja0) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    ja0_nw = (3 * counters["total_w_sim"]) / (3 * counters["total_sim"] + counters["total_dis"])
    return ja0_nw


def _jt_nw(counters: dict) -> float:
    """Calculate the Jaccard-Tanimoto (JT) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    jt_nw = (counters["w_a"]) / (counters["a"] + counters["total_dis"])
    return jt_nw


def _rt_nw(counters: dict) -> float:
    """Calculate the Rogers-Tanimoto (RT) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    rt_nw = (counters["total_w_sim"]) / (counters["p"] + counters["total_dis"])
    return rt_nw


def _rr_nw(counters: dict) -> float:
    """Calculate the Russel-Rao (RR) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    rr_nw = (counters["w_a"]) / (counters["p"])
    return rr_nw


def _sm_nw(counters: dict) -> float:
    """Calculate the Sokal-Michener (SM) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    sm_nw = (counters["total_w_sim"]) / (counters["p"])
    return sm_nw


def _ss1_nw(counters: dict) -> float:
    """Calculate the Sokal-Sneath 1 (SS1) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    ss1_nw = (counters["w_a"]) / (counters["a"] + 2 * counters["total_dis"])
    return ss1_nw


def _ss2_nw(counters: dict) -> float:
    """Calculate the Sokal-Sneath 2 (SS2) similarity index.

    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3"""
    ss2_nw = (2 * counters["total_w_sim"]) / (counters["p"] + counters["total_sim"])
    return ss2_nw


# Dictionary with the similarity indexes functions as values and the keys are the abbreviations
_similarity_index_dict = {
    "AC": _ac_nw,
    "BUB": _bub_nw,
    "CT1": _ct1_nw,
    "CT2": _ct2_nw,
    "CT3": _ct3_nw,
    "CT4": _ct4_nw,
    "Fai": _fai_nw,
    "Gle": _gle_nw,
    "Ja": _ja_nw,
    "Ja0": _ja0_nw,
    "JT": _jt_nw,
    "RT": _rt_nw,
    "RR": _rr_nw,
    "SM": _sm_nw,
    "SS1": _ss1_nw,
    "SS2": _ss2_nw,
}
