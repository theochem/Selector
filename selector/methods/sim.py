# The Selector library provides a set of tools for selecting a
# subset of the dataset and computing diversity.
#
# Copyright (C) 2023 The QC-Devs Community
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
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00505-3
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00504-4

"""

import random
from typing import List, Optional, Union

import numpy as np

from selector.methods.base import SelectionBase
from selector.similarity import SimilarityIndex

__all__ = ["NSimilarity"]


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
            if not isinstance(start, list) or not all(isinstance(i, int) for i in start):
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
