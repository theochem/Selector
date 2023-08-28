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

import math
from math import log
import random
from typing import List, Optional, Union

from DiverseSelector.methods.base import SelectionBase
import numpy as np


__all__ = ["NSimilarity", "SimilarityIndex"]


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
        https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3
        https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00504-4
        https://link.springer.com/article/10.1007/s10822-022-00444-7
    """

    def __init__(
        self,
        similarity_index: str = "RR",
        w_factor: str = "fraction",
        c_threshold: Union[None, str, int] = None,
        preprocess_data: bool = True,
    ):
        """Initialize class.

        Parameters
        ----------
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
        c_threshold: {None, 'dissimilar', int}
            Coincidence threshold used for calculating the similarity counters. A column of the
            elements is considered to be a coincidence among the elements if the number of elements
            that have the same value in that position is greater than the coincidence threshold.
                None : Default, c_threshold = n_objects % 2
                'dissimilar' : c_threshold = ceil(n_objects / 2)
                int : Integer number < n_objects
        preprocess_data: bool
            Every data element must be betwen 0 and 1 for the similarity indexes to work. If
            preprocess_data is True, the data is scaled between 0 and 1 using a strategy that is
            compatible with the similarity indexes. If preprocess_data is False, the data is not
            scaled and it is assumed that the data is already between 0 and 1.

        """
        # check if the similarity index is valid
        if similarity_index not in _similarity_index_dict:
            raise ValueError(
                f'Similarity index "{similarity_index}" is not available. '
                f"See the documentation for the available similarity indexes."
            )
        # check if the w_factor is valid
        if w_factor not in ["fraction", "power_n"]:
            print(
                f'Weight factor "{w_factor}" given. Using default value '
                '"similarity = dissimilarity = 1".'
            )
        # check if the c_threshold is valid
        if c_threshold not in ["dissimilar", None]:
            if not isinstance(c_threshold, int):
                raise ValueError(
                    f'Invalid c_threshold. It must be an integer or "dissimilar" or None. '
                    f"Given c_threshold = {c_threshold}"
                )

        self.similarity_index = similarity_index
        self.w_factor = w_factor
        self.c_threshold = c_threshold
        self.preprocess_data = preprocess_data

    def _scale_data(self, arr: np.ndarray):
        r"""Scales the data between so it can be used with the similarity indexes.

        First each data point is normalized to be between 0 and 1.
        .. math::
            x_{ij} = \\frac{x_{ij} - min(x_j)}{max(x_j) - min(x_j)}

        Then, the average of each column is calculated. Finally, each element of the final working
        array will be defined as

        .. math::
            w_ij = 1 - | x_ij - a_j |

        where $x_ij$ is the element of the normalized array, and $a_j$ is the average of the j-th
        column of the normalized array.

        Parameters
        ----------
        arr: np.ndarray
            Array of features (columns) for each sample (rows).
        """
        min_value = np.min(arr)
        max_value = np.max(arr)
        # normalize the data to be between 0 and 1 for working with the similarity indexes
        normalized_data = (arr - min_value) / (max_value - min_value)
        # calculate the average of the columns
        col_average = np.average(normalized_data, axis=0)

        # each element of the final working array will be defined as w_ij = 1 - | x_ij - a_j |
        # where x_ij is the element of the normalized array, and a_j is the average of the j-th
        # column of the normalized array.
        data = 1 - np.abs(normalized_data - col_average)
        return data

    def _get_new_index(
        self,
        arr: np.ndarray,
        selected_condensed: np.ndarray,
        num_selected: int,
        select_from: np.ndarray,
    ) -> int:
        r"""Select a new diverse molecule from the data.

        The function selects a new molecule such that the similarity of the new set of selected
        molecules is minimized.

        Parameters
        ----------
        arr: np.ndarray
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
        if np.max(arr) > 1 or np.min(arr) < 0:
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
        index = arr.shape[0] + 1

        # create an instance of the SimilarityIndex class. It is used to calculate the similarity
        # index of the sets of selected objects.
        similarity_index = SimilarityIndex(
            similarity_index=self.similarity_index,
            c_threshold=self.c_threshold,
            w_factor=self.w_factor,
        )

        # for all indices that have not been selected
        for sample_idx in select_from:
            # column sum
            c_total = selected_condensed + arr[sample_idx]

            # calculating similarity
            sim_index = similarity_index(c_total, n_objects=n_total)

            # if the sim of the set is less than the similarity of the previous diverse set,
            # update min_value and index
            if sim_index < min_value:
                index = sample_idx
                min_value = sim_index

        return index

    def select_from_cluster(
        self,
        arr: np.ndarray,
        size: int,
        cluster_ids: Optional[np.ndarray] = None,
        start: Union[str, List[int]] = "medoid",
    ) -> List[int]:
        r"""Algorithm of nary similarity selection for selecting points from cluster.

        Parameters
        ----------
        arr: np.ndarray
            Array of features (columns) for each sample (rows).
        size: int
            Number of sample points to select (i.e. size of the subset).
        cluster_ids: np.ndarray, optional
            Array of integers or strings representing the points ids of the data that belong to the
            current cluster. If `None`, all the samples in the data are treated as one cluster.
        start: str or list
            srt: key on what is used to start the selection
                {'medoid', 'random', 'outlier'}
            list: indices of points that are included in the selection since the beginning

        Returns
        -------
        selected: list
            Indices of the selected sample points.
        """
        # check for valid start value and raise an error if it is not
        if start not in ["medoid", "random", "outlier"]:
            if not all(isinstance(i, int) for i in start):
                raise ValueError(
                    "Select a correct starting point: medoid, random, outlier or a list of indices."
                )
        # check if cluster_ids are provided
        if cluster_ids is not None:
            # extract the data corresponding to the cluster_ids
            arr = np.take(arr, cluster_ids, axis=0)

        # total number of objects in the current cluster
        samples = arr.shape[0]
        # ids of the data points in the current cluster (0, 1, 2, ..., samples-1)
        data_ids = np.array(range(samples))

        # check if the number of selected objects is less than the total number of objects
        if size > samples:
            raise ValueError(
                f"Number of samples is less than the requested sample size: {samples} < {size}."
            )

        # The data is marked to be preprocessed scale the data between 0 and 1 using a strategy
        # that is compatible with the similarity indexes
        if self.preprocess_data:
            arr = self._scale_data(arr)
        else:
            # check if the data is between 0 and 1 and raise an error if it is not
            if np.max(arr) > 1 or np.min(arr) < 0:
                raise ValueError(
                    "The data was not scaled between 0 and 1. "
                    "Use the _scale_data function to scale the data."
                )

        # create an instance of the SimilarityIndex class. It is used to calculate the medoid and
        # the outlier of the data.
        similarity_index = SimilarityIndex(
            similarity_index=self.similarity_index,
            c_threshold=self.c_threshold,
            w_factor=self.w_factor,
        )

        # select the index (of the working data) corresponding to the medoid of the data using the
        # similarity index
        if start == "medoid":
            seed = similarity_index.calculate_medoid(arr)
            selected = [seed]
        # select the index (of the working data)  corresponding to a random data point
        elif start == "random":
            seed = random.randint(0, samples - 1)
            selected = [seed]
        # select the index (of the working data) corresponding to the outlier of the data using the
        # similarity index
        elif start == "outlier":
            seed = similarity_index.calculate_outlier(arr)
            selected = [seed]
        # if a list of cluster_ids is provided, select the data_ids corresponding indices
        elif isinstance(start, list):
            if cluster_ids is not None:
                # check if all starting indices are in this cluster
                if not all(label in cluster_ids for label in start):
                    raise ValueError(
                        "Some of the provided initial indexes are not in the cluster data."
                    )
                # select the indices of the data_ids that correspond to the provided starting points
                # provided from cluster_ids
                selected = [i for i, j in enumerate(cluster_ids) if j in start]
            else:
                # check if all starting indices are in the data
                if not all(label in data_ids for label in start):
                    raise ValueError("Some of the provided initial indexes are not in the data.")
                # select the indices of the data_ids that correspond to the provided starting points
                selected = start[:]
        else:
            raise ValueError(
                "Select a correct starting point: medoid, random, outlier or a list of indices"
            )
        # Number of initial objects
        num_selected = len(selected)

        # get selected samples form the working data array
        selected_objects = np.take(arr, selected, axis=0)
        # Calculate the columnwise sum of the selected samples
        selected_condensed = np.sum(selected_objects, axis=0)

        # until the desired number of objects is selected a new object is selected.
        while num_selected < size:
            # indices from which to select the new data points
            select_from = np.delete(data_ids, selected)

            # Select new index. The new object is selected such that from all possible objects the
            # similarity of the set of (selected_objects + new_object) is a minimum.
            new_index = self._get_new_index(arr, selected_condensed, num_selected, select_from)

            # updating column sum vector
            selected_condensed += arr[new_index]

            # updating selected indices
            selected.append(new_index)
            num_selected += 1

        return selected


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
        Calculate the outlier of a set of real-valued vectors or binary objects. The
        similarity_index is used as the distance.

    __call__(data=None, n_objects=None, similarity_index=None, c_threshold=None, w_factor=None):
        Calculate the similarity index of a set of vectors.

    """

    def __init__(
        self,
        similarity_index: str = "RR",
        c_threshold: Union[None, str, int] = None,
        w_factor: str = "fraction",
    ):
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
            The coincidence threshold used for calculating similarity counters. A position in the
            elements is considered a coincidence (coincides among all the elements considered) if
            the number of elements that have the same value in that position is greater than the
            coincidence threshold.
                - None : Default, c_threshold = n_objects % 2
                - 'dissimilar' : c_threshold = ceil(n_objects / 2)
                - int : Integer number < n_objects

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
        """
        # check if the similarity index is valid
        if similarity_index not in _similarity_index_dict:
            raise ValueError(
                f'Similarity index "{similarity_index}" is not available. '
                f"See the documentation for the available similarity indexes."
            )
        # check if the c_threshold is valid
        if c_threshold not in ["dissimilar", None]:
            if not isinstance(c_threshold, int):
                raise ValueError(
                    f'Invalid c_threshold. It must be an integer or "dissimilar" or None. '
                    f"Given c_threshold = {c_threshold}"
                )
        # check if the w_factor is valid
        if w_factor not in ["fraction", "power_n"]:
            print(
                f'Weight factor "{w_factor}" given. Using default value '
                '"similarity = dissimilarity = 1".'
            )

        self.similarity_index = similarity_index
        self.w_factor = w_factor
        self.c_threshold = c_threshold

    def _calculate_counters(
        self,
        arr: np.ndarray = None,
        n_objects: Optional[int] = None,
        c_threshold: Union[None, str, int] = None,
        w_factor: Optional[str] = None,
    ) -> dict:
        """Calculate 1-similarity, 0-similarity, and dissimilarity counters.

        Arguments
        ---------
        arr : np.ndarray
            Array of arrays, each sub-array contains the binary or real valued vector. The values
            must be between 0 and 1. If the number of rows ==1, the data is treated as the
            columnwise sum of the objects. If the number of rows > 1, the data is treated as the
            objects.
        n_objects: int
            Number of objects, only necessary if the columnwise sum of the objects is provided
            instead of the data (num rows== 1). If the data is provided, the number of objects is
            calculated as the length of the data.
        c_threshold: {None, 'dissimilar', int}
            Coincidence threshold used for calculating the similarity counters. A column of the
            elements is considered to be a coincidence among the elements if the number of elements
            that have the same value in that position is greater than the coincidence threshold.
                None : Default, c_threshold = n_objects % 2
                'dissimilar' : c_threshold = ceil(n_objects / 2)
                int : Integer number < n_objects
        w_factor: {"fraction", "power_n"}
            Type of weight function that will be used.
            'fraction' : similarity = d[k]/n
                            dissimilarity = 1 - (d[k] - n_objects % 2)/n_objects
            'power_n' : similarity = n**-(n_objects - d[k])
                        dissimilarity = n**-(d[k] - n_objects % 2)
            other values : similarity = dissimilarity = 1

        Returns
        -------
        counters : dict
            Dictionary with the weighted and non-weighted counters.
        """
        # Check if the data is a np.ndarray of a list
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                "Input data is not a np.ndarray, to secure the right results please "
                "input the right data type"
            )

        # Check if data is a columnwise sum or the objects
        if arr.ndim == 1:
            c_total = arr
            # If data is a columnwise sum, check if n_objects is provided
            if n_objects is None:
                raise ValueError(
                    "Input data is the columnwise sum, please specify number of objects"
                )
        else:
            c_total = np.sum(arr, axis=0)
            n_objects = len(arr)

        # Assign c_threshold
        if not c_threshold:
            c_threshold = n_objects % 2
        elif c_threshold == "dissimilar":
            c_threshold = math.ceil(n_objects / 2)
        elif isinstance(c_threshold, int):
            if c_threshold >= n_objects:
                raise ValueError(
                    "c_threshold cannot be equal or greater than n_objects. \n"
                    + f"c_threshold = {c_threshold}  n_objects = {n_objects}"
                )
            c_threshold = c_threshold
        else:
            raise ValueError(
                "c_threshold must be None, 'dissimilar' or an integer. \n"
                + f"Given c_threshold = {c_threshold}"
            )

        # Set w_factor function (a weight factor for the similarity and dissimilarity) is
        # provided depending on the number of objects that are similar or not in a column
        if w_factor:
            # power_n case
            if "power" in w_factor:
                power = int(w_factor.split("_")[-1])

                def f_s(d):
                    return power ** -(n_objects - d).astype(float)

                def f_d(d):
                    return power ** -(d - n_objects % 2).astype(float)

            # fraction case
            elif w_factor == "fraction":

                def f_s(d):
                    return d / n_objects

                def f_d(d):
                    return 1 - (d - n_objects % 2) / n_objects

            else:
                raise ValueError(
                    f"w_factor must be 'fraction' or 'power_n'. \n Given w_factor = {w_factor}"
                )
        # default case, the similarity and dissimilarity counters are not weighted
        else:

            def f_s(d):
                return 1

            def f_d(d):
                return 1

        # Calculate a, d, b + c
        # Calculate the positions (columns) of common on bits (common 1s) between the objects
        a_indices = 2 * c_total - n_objects > c_threshold
        # Calculate the positions (columns) common off bits (common 0s) between the objects
        d_indices = n_objects - 2 * c_total > c_threshold
        # Calculate the positions (columns) of dissimilar bits between the objects (b + c)
        # the dissimilar bits are the bits that are not common between the objects
        dis_indices = np.abs(2 * c_total - n_objects) <= c_threshold

        # Calculate the number of columns with common on bits (common 1s) between the objects
        a = np.sum(a_indices)
        # Calculate the number of columns with common off bits (common 0s) between the objects
        d = np.sum(d_indices)
        # Calculate the number of columns with dissimilar bits between the objects (b + c)
        total_dis = np.sum(dis_indices)

        # calculate the weights for each column indexed as with common on bits (common 1s)
        a_w_array = f_s(2 * c_total[a_indices] - n_objects)
        # calculate the weights for each column indexed as with common off bits (common 0s)
        d_w_array = f_s(abs(2 * c_total[d_indices] - n_objects))
        # calculate the weights for each column indexed as with dissimilar bits
        total_w_dis_array = f_d(abs(2 * c_total[dis_indices] - n_objects))

        # calculate the total weight for each type of counter
        w_a = np.sum(a_w_array)
        w_d = np.sum(d_w_array)
        total_w_dis = np.sum(total_w_dis_array)

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

    def __call__(
        self,
        data: np.ndarray = None,
        n_objects: int = None,
        similarity_index: str = None,
        c_threshold: Union[None, str, int] = None,
        w_factor: str = None,
    ) -> float:
        """Calculate the similarity index of a set of vectors.

        Parameters
        ----------
        data : np.ndarray
            Array of arrays, each sub-array contains the binary or real valued vector. The values
            must be between 0 and 1. If the number of rows ==1, the data is treated as the
            columnwise sum of the objects. If the number of rows > 1, the data is treated as the
            objects.
        n_objects: int
            Number of objects in the data. Is only necessary if the data is a columnwise sum of
            the objects. If the data is not the columnwise sum of the objects, the number of objects
            is calculated as the length of the data.
        similarity_index: string
            Key with the abbreviation of the desired similarity index to be calculated for the data.
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
        c_threshold: {None, 'dissimilar', int}
            Coincidence threshold used for calculating the similarity counters. A position of the
            elements is considered to be a coincidence (coincides among all the elements considered)
            if the number of elements that have the same value in that position is greater than the
            coincidence threshold.
            arrays
                None : Default, c_threshold = n_objects % 2
                'dissimilar' : c_threshold = ceil(n_objects / 2)
                int : Integer number < n_objects
        w_factor: {"fraction", "power_n"}
            Type of weight function that will be used.
            'fraction' : similarity = d[k]/n
                            dissimilarity = 1 - (d[k] - n_objects % 2)/n_objects
            'power_n' : similarity = n**-(n_objects - d[k])
                        dissimilarity = n**-(d[k] - n_objects % 2)
            other values : similarity = dissimilarity = 1

        Returns
        -------
        similarity_index: float
            Similarity index of the set of vectors.
        """
        # If the parameters are not provided, the parameters provided in the class initialization
        # are used. If the parameters are provided, the parameters values are checked and used.
        if similarity_index is None:
            similarity_index = self.similarity_index
        else:
            # check if the similarity index is valid
            if similarity_index not in _similarity_index_dict:
                raise ValueError(
                    f'Similarity index "{similarity_index}" is not available. '
                    f"See the documentation for the available similarity indexes."
                )
        if w_factor is None:
            w_factor = self.w_factor
        else:
            # check if the w_factor is valid
            if w_factor not in ["fraction", "power_n"]:
                print(
                    f'Weight factor "{w_factor}" given. Using default value '
                    '"similarity = dissimilarity = 1".'
                )
        if c_threshold is None:
            c_threshold = self.c_threshold
        else:
            # check if the c_threshold is valid
            if c_threshold not in ["dissimilar", None]:
                if not isinstance(c_threshold, int):
                    raise ValueError(
                        f'Invalid c_threshold. It must be an integer or "dissimilar" or None. '
                        f"Given c_threshold = {c_threshold}"
                    )

        # check that data or c_total is provided
        if data is None:
            raise ValueError("Please provide data or c_total")

        # check if data is a np.ndarray
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "Warning: Input data is not a np.ndarray, to secure the right results please input "
                "the right data type"
            )

        # if the data is a columnwise sum of the objects check that n_objects is provided
        if data.ndim == 1:
            c_total = data
            if not n_objects:
                raise ValueError(
                    "Input data is the columnwise sum, please specify number of objects"
                )
        # if the data is not a columnwise sum of the objects, calculate the columnwise sum and the
        # number of objects
        else:
            c_total = np.sum(data, axis=0)
            n_objects = data.shape[0]

        # calculate the counters needed to calculate the similarity indexes
        counters = self._calculate_counters(
            arr=c_total, n_objects=n_objects, w_factor=w_factor, c_threshold=c_threshold
        )
        # calculate the similarity index
        similarity_index = _similarity_index_dict[similarity_index](counters)

        return similarity_index

    def calculate_medoid(
        self,
        data: np.ndarray = None,
        c_total=None,
        similarity_index: str = None,
        c_threshold=None,
        w_factor: str = None,
    ) -> int:
        """Calculate the medoid of a set of real-valued vectors or binary objects.

        Parameters
        ----------
        data: np.array
            np.array of all the real-valued vectors or binary objects.
        c_total:
            np.array with the columnwise sums of the data, not necessary to provide.
        similarity_index: string
            Key with the abbreviation of the desired similarity index to calculate the medoid from.
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
        c_threshold: {None, 'dissimilar', int}
            Coincidence threshold used for calculating the similarity counters. A column of the
            elements is considered to be a coincidence among the elements if the number of elements
            that have the same value in that position is greater than the coincidence threshold.
                None : Default, c_threshold = n_objects % 2
                'dissimilar' : c_threshold = ceil(n_objects / 2)
                int : Integer number < n_objects
        w_factor: {"fraction", "power_n"}
            Type of weight function that will be used.
            'fraction' : similarity = d[k]/n
                            dissimilarity = 1 - (d[k] - n_objects % 2)/n_objects
            'power_n' : similarity = n**-(n_objects - d[k])
                        dissimilarity = n**-(d[k] - n_objects % 2)
            other values : similarity = dissimilarity = 1
        """
        # If the parameters are not provided, the parameters provided in the class initialization
        # are used. If the parameters are provided, the parameters values are checked and used.
        if similarity_index is None:
            similarity_index = self.similarity_index
        else:
            # check if the similarity index is valid
            if similarity_index not in _similarity_index_dict:
                raise ValueError(
                    f'Similarity index "{similarity_index}" is not available. '
                    f"See the documentation for the available similarity indexes."
                )
        if w_factor is None:
            w_factor = self.w_factor
        else:
            # check if the w_factor is valid
            if w_factor not in ["fraction", "power_n"]:
                print(
                    f'Weight factor "{w_factor}" given. Using default value '
                    '"similarity = dissimilarity = 1".'
                )
        if c_threshold is None:
            c_threshold = self.c_threshold
        else:
            # check if the c_threshold is valid
            if c_threshold not in ["dissimilar", None]:
                if not isinstance(c_threshold, int):
                    raise ValueError(
                        f'Invalid c_threshold. It must be an integer or "dissimilar" or None. '
                        f"Given c_threshold = {c_threshold}"
                    )

        # check if c_total is provided and if not, calculate it
        if c_total is None:
            c_total = np.sum(data, axis=0)
        # if c_total is provided, check if it has the same number of columns as the data
        elif c_total is not None and len(data[0]) != len(c_total):
            raise ValueError("Dimensions of objects and columnwise sum differ")

        # get the total number of objects
        n_objects = data.shape[0]

        # Initialize the selected index with a number outside the possible index values
        index = n_objects + 1

        # minimum similarity value that is guaranteed to be higher than all the comparisons, this
        # value should be a warranty that a exist a sample with similarity lower than min_sim. The
        # max possible similarity value for set of samples is 1.00.
        min_sim = 1.01

        # For each sample in the set, calculate the columnwise sum of the data without the sample
        comp_sums = c_total - data

        # for each sample calculate the similarity index of the complete set without the sample
        for idx, obj in enumerate(comp_sums):
            # calculate the similarity index of the set of objects without the current object
            sim_index = self(
                data=obj,
                n_objects=n_objects - 1,
                similarity_index=similarity_index,
                w_factor=w_factor,
                c_threshold=c_threshold,
            )
            # if the similarity is lower than the previous minimum similarity, update the minimum
            # similarity and the index
            if sim_index < min_sim:
                min_sim, index = sim_index, idx
            else:
                pass
        # the index of the object that increases more the similarity of the set when added is
        # returned (the medoid)
        return index

    def calculate_outlier(
        self,
        data: np.ndarray = None,
        c_total=None,
        similarity_index: str = None,
        c_threshold=None,
        w_factor: str = None,
    ) -> int:
        r"""Calculate the outlier of a set of real-valued vectors or binary objects.

        Calculates the outlier of a set of real-valued vectors or binary objects. Using the
        similarity index provided in the class initialization.

        Parameters
        ----------
        data: np.array
            np.array of all the real-valued vectors or binary objects.
        c_total:
            np.array with the columnwise sums of the data, not necessary to provide.
        similarity_index: string
            Key with the abbreviation of the desired similarity index to calculate the medoid from.
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
        c_threshold: {None, 'dissimilar', int}
            Coincidence threshold used for calculating the similarity counters. A column of the
            elements is considered to be a coincidence among the elements if the number of elements
            that have the same value in that position is greater than the coincidence threshold.
                None : Default, c_threshold = n_objects % 2
                'dissimilar' : c_threshold = ceil(n_objects / 2)
                int : Integer number < n_objects
        w_factor: {"fraction", "power_n"}
            Type of weight function that will be used.
            'fraction' : similarity = d[k]/n
                            dissimilarity = 1 - (d[k] - n_objects % 2)/n_objects
            'power_n' : similarity = n**-(n_objects - d[k])
                        dissimilarity = n**-(d[k] - n_objects % 2)
            other values : similarity = dissimilarity = 1
        """
        # If the parameters are not provided, the parameters provided in the class initialization
        # are used. If the parameters are provided, the parameters values are checked and used.
        if similarity_index is None:
            similarity_index = self.similarity_index
        else:
            # check if the similarity index is valid
            if similarity_index not in _similarity_index_dict:
                raise ValueError(
                    f'Similarity index "{similarity_index}" is not available. '
                    f"See the documentation for the available similarity indexes."
                )
        if w_factor is None:
            w_factor = self.w_factor
        else:
            # check if the w_factor is valid
            if w_factor not in ["fraction", "power_n"]:
                print(
                    f'Weight factor "{w_factor}" given. Using default value '
                    '"similarity = dissimilarity = 1".'
                )
        if c_threshold is None:
            c_threshold = self.c_threshold
        else:
            # check if the c_threshold is valid
            if c_threshold not in ["dissimilar", None]:
                if not isinstance(c_threshold, int):
                    raise ValueError(
                        f'Invalid c_threshold. It must be an integer or "dissimilar" or None. '
                        f"Given c_threshold = {c_threshold}"
                    )

        # check if c_total is provided and if not, calculate it
        if c_total is None:
            c_total = np.sum(data, axis=0)
        # if c_total is provided, check if it has the same number of columns as the data
        elif c_total is not None and len(data[0]) != len(c_total):
            raise ValueError("Dimensions of objects and columnwise sum differ")

        n_objects = data.shape[0]

        # Initialize the selected index with a number outside the possible index values
        index = n_objects + 1

        # maximum similarity value that is guaranteed to be lower than all the comparisons, this
        # value should be a warranty that a exist a sample with similarity lower than min_sim. The
        # min possible similarity value for set of samples is 0.00.
        max_sim = -0.01

        # For each sample in the set, calculate the columnwise sum of the data without the sample
        comp_sums = c_total - data

        # for each sample calculate the similarity index of the complete set without the sample
        for idx, obj in enumerate(comp_sums):
            # calculate the similarity index of the set of objects without the current object
            sim_index = self.__call__(
                data=obj,
                n_objects=n_objects - 1,
                similarity_index=similarity_index,
                w_factor=w_factor,
                c_threshold=c_threshold,
            )
            # if the similarity is bigger than the previous minimum similarity, update the minimum
            # similarity and the index
            if sim_index > max_sim:
                max_sim, index = sim_index, idx
            else:
                pass
        # the index of the object that decreases more the similarity of the set when added is
        # returned (the outlier)
        return index


# Utility functions section
# -------------------------

# Functions that calculate the similarity indexes. The functions are named as the similarity
# index they calculate. The _nw suffix indicates that the similarity index is not weighted.
# More information about the similarity indexes can be found in the following paper:
# https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3


# AC: Austin-Colwell
def _ac_nw(counters: dict) -> float:
    ac_nw = (2 / np.pi) * np.arcsin(np.sqrt(counters["total_w_sim"] / counters["p"]))
    return ac_nw


# BUB: Baroni-Urbani-Buser
def _bub_nw(counters: dict) -> float:
    bub_nw = ((counters["w_a"] * counters["w_d"]) ** 0.5 + counters["w_a"]) / (
        (counters["a"] * counters["d"]) ** 0.5 + counters["a"] + counters["total_dis"]
    )
    return bub_nw


# CTn: Consoni-Todschini 1
def _ct1_nw(counters: dict) -> float:
    ct1_nw = (log(1 + counters["w_a"] + counters["w_d"])) / (log(1 + counters["p"]))
    return ct1_nw


# CTn: Consoni-Todschini 2
def _ct2_nw(counters: dict) -> float:
    ct2_nw = (log(1 + counters["w_p"]) - log(1 + counters["total_w_dis"])) / (
        log(1 + counters["p"])
    )
    return ct2_nw


# CTn: Consoni-Todschini 3
def _ct3_nw(counters: dict) -> float:
    ct3_nw = (log(1 + counters["w_a"])) / (log(1 + counters["p"]))
    return ct3_nw


# CTn: Consoni-Todschini 4
def _ct4_nw(counters: dict) -> float:
    ct4_nw = (log(1 + counters["w_a"])) / (log(1 + counters["a"] + counters["total_dis"]))
    return ct4_nw


# Fai: Faith
def _fai_nw(counters: dict) -> float:
    fai_nw = (counters["w_a"] + 0.5 * counters["w_d"]) / (counters["p"])
    return fai_nw


# Gle: Gleason
def _gle_nw(counters: dict) -> float:
    gle_nw = (2 * counters["w_a"]) / (2 * counters["a"] + counters["total_dis"])
    return gle_nw


# Ja: Jaccard
def _ja_nw(counters: dict) -> float:
    ja_nw = (3 * counters["w_a"]) / (3 * counters["a"] + counters["total_dis"])
    return ja_nw


# Ja0: Jaccard 0-variant
def _ja0_nw(counters: dict) -> float:
    ja0_nw = (3 * counters["total_w_sim"]) / (3 * counters["total_sim"] + counters["total_dis"])
    return ja0_nw


# JT: Jaccard-Tanimoto
def _jt_nw(counters: dict) -> float:
    jt_nw = (counters["w_a"]) / (counters["a"] + counters["total_dis"])
    return jt_nw


# RT: Rogers-Tanimoto
def _rt_nw(counters: dict) -> float:
    rt_nw = (counters["total_w_sim"]) / (counters["p"] + counters["total_dis"])
    return rt_nw


# RR: Russel-Rao
def _rr_nw(counters: dict) -> float:
    rr_nw = (counters["w_a"]) / (counters["p"])
    return rr_nw


# SM: Sokal-Michener
def _sm_nw(counters: dict) -> float:
    sm_nw = (counters["total_w_sim"]) / (counters["p"])
    return sm_nw


# SSn: Sokal-Sneath 1
def _ss1_nw(counters: dict) -> float:
    ss1_nw = (counters["w_a"]) / (counters["a"] + 2 * counters["total_dis"])
    return ss1_nw


# SSn: Sokal-Sneath 2
def _ss2_nw(counters: dict) -> float:
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
