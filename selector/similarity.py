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
"""Similarity Module."""

import math
import warnings
from itertools import combinations_with_replacement
from math import log
from typing import Optional, Union

import numpy as np

__all__ = [
    "pairwise_similarity_bit",
    "tanimoto",
    "modified_tanimoto",
    "SimilarityIndex",
]


def pairwise_similarity_bit(X: np.array, metric: str) -> np.ndarray:
    """Compute pairwise similarity coefficient matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix of `n_samples` samples in `n_features` dimensional space.
    metric : str
        The metric used when calculating similarity coefficients between samples in a feature array.
        Method for calculating similarity coefficient. Options: `"tanimoto"`, `"modified_tanimoto"`.

    Returns
    -------
    s : ndarray of shape (n_samples, n_samples)
        A symmetric similarity matrix between each pair of samples in the feature matrix.
        The diagonal elements are directly computed instead of assuming that they are 1.
    """

    available_methods = {
        "tanimoto": tanimoto,
        "modified_tanimoto": modified_tanimoto,
    }
    if metric not in available_methods:
        raise ValueError(
            f"Argument metric={metric} is not recognized! Choose from {available_methods.keys()}"
        )
    if X.ndim != 2:
        raise ValueError(f"Argument features should be a 2D array, got {X.ndim}")

    # make pairwise m-by-m similarity matrix
    n_samples = len(X)
    s = np.zeros((n_samples, n_samples))
    # compute similarity between all pairs of points (including the diagonal elements)
    for i, j in combinations_with_replacement(range(n_samples), 2):
        s[i, j] = s[j, i] = available_methods[metric](X[i], X[j])
    return s


class SimilarityIndex:
    r"""Calculate the n-ary similarity index of a set of vectors.

    This class provides methods for calculating the similarity index of a set of vectors represented
    as a matrix. Each vector is a row in the matrix, and each column represents a feature of the
    vector. The features in the vectors must be binary or real numbers between 0 and 1.

    Methods
    -------
    calculate_medoid(arr, c_total=None):
        Calculate the medoid of a set of real-valued vectors or binary objects. The similarity_index
        is used as the distance.

    calculate_outlier(arr, c_total=None):
        Calculate the outlier of a set of real-valued vectors or binary objects. The
        similarity_index is used as the distance.

    __call__(arr=None, n_objects=None):
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
        if w_factor != "fraction":
            if w_factor.split("_")[0] != "power" or not w_factor.split("_")[-1].isdigit():
                warnings.warn(
                    f'Invalid weight factor "{w_factor}" given. Using default value '
                    '"similarity = dissimilarity = 1".'
                )
                w_factor = False

        self.similarity_index = similarity_index
        self.w_factor = w_factor
        self.c_threshold = c_threshold

    def _calculate_counters(self, arr: np.ndarray, n_objects: Optional[int] = None) -> dict:
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
        if self.c_threshold is None:
            tmp_c_threshold = n_objects % 2
        elif self.c_threshold == "dissimilar":
            tmp_c_threshold = math.ceil(n_objects / 2)
        elif isinstance(self.c_threshold, int):
            if self.c_threshold >= n_objects:
                raise ValueError(
                    "c_threshold cannot be equal or greater than n_objects. \n"
                    f"c_threshold = {self.c_threshold}  n_objects = {n_objects}"
                )
            tmp_c_threshold = self.c_threshold
        else:
            raise ValueError(
                "c_threshold must be None, 'dissimilar' or an integer. \n"
                f"Given c_threshold = {self.c_threshold}"
            )

        # Calculate a, d, b + c
        # Calculate the positions (columns) of common on bits (common 1s) between the objects
        a_indices = 2 * c_total - n_objects > tmp_c_threshold
        # Calculate the positions (columns) common off bits (common 0s) between the objects
        d_indices = n_objects - 2 * c_total > tmp_c_threshold
        # Calculate the positions (columns) of dissimilar bits between the objects (b + c)
        # the dissimilar bits are the bits that are not common between the objects
        dis_indices = np.abs(2 * c_total - n_objects) <= tmp_c_threshold

        # Calculate the number of columns with common on bits (common 1s) between the objects
        a = np.sum(a_indices)
        # Calculate the number of columns with common off bits (common 0s) between the objects
        d = np.sum(d_indices)
        # Calculate the number of columns with dissimilar bits between the objects (b + c)
        total_dis = np.sum(dis_indices)

        # calculate the weights for each column indexed as with common on bits (common 1s)
        a_w_array = self._f_s(2 * c_total[a_indices] - n_objects, n_objects)
        # calculate the weights for each column indexed as with common off bits (common 0s)
        d_w_array = self._f_s(abs(2 * c_total[d_indices] - n_objects), n_objects)
        # calculate the weights for each column indexed as with dissimilar bits
        total_w_dis_array = self._f_d(abs(2 * c_total[dis_indices] - n_objects), n_objects)

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
        else:
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
        else:
            return 1

    def __call__(self, arr: np.ndarray, n_objects: int = None) -> float:
        """Calculate the similarity index of a set of vectors.

        Parameters
        ----------
        arr : np.ndarray
            Array of arrays, each sub-array contains the binary or real valued vector. The values
            must be between 0 and 1. If the number of rows ==1, the data is treated as the
            columnwise sum of the objects. If the number of rows > 1, the data is treated as the
            objects.
        n_objects: int
            Number of objects in the data. Is only necessary if the data is a columnwise sum of
            the objects. If the data is not the columnwise sum of the objects, the number of objects
            is calculated as the length of the data.
        Returns
        -------
        similarity_index: float
            Similarity index of the set of vectors.
        """
        # check if arr is a np.ndarray
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input data is not a np.ndarray, please input the right data type")

        # if the data is a columnwise sum of the objects check that n_objects is provided
        if arr.ndim == 1:
            c_total = arr
            if not n_objects:
                raise ValueError(
                    "Input data is the columnwise sum, please specify number of objects"
                )
        # if the data is not a columnwise sum of the objects, calculate the columnwise sum and the
        # number of objects
        else:
            c_total = np.sum(arr, axis=0)
            n_objects = arr.shape[0]

        # calculate the counters needed to calculate the similarity indexes
        counters = self._calculate_counters(arr=c_total, n_objects=n_objects)
        # calculate the similarity index
        similarity_index = _similarity_index_dict[self.similarity_index](counters)

        return similarity_index

    def calculate_medoid(self, arr: np.ndarray, c_total=None) -> int:
        """Calculate the medoid of a set of real-valued vectors or binary objects.

        Parameters
        ----------
        arr: np.array
            np.array of all the real-valued vectors or binary objects.
        c_total:
            np.array with the columnwise sums of the data, not necessary to provide.
        """
        # Check if the data is a np.ndarray of a list
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input data is not a np.ndarray, please input the right data type")
        # Check if the data is one dimensional
        if arr.ndim != 2:
            raise ValueError(
                "Data must be a two dimensional np.ndarray for calculating the medoid."
            )
        # Check if the data has at least 3 rows
        if arr.shape[0] < 3:
            raise ValueError("Input data must have at least 3 rows to calculate the medoid.")

        # check if c_total is provided and if not, calculate it
        if c_total is None:
            c_total = np.sum(arr, axis=0)
        # if c_total is provided, check if it has the same number of columns as the data
        elif c_total is not None and len(arr[0]) != len(c_total):
            raise ValueError("Dimensions of objects and columnwise sum differ")

        # get the total number of objects
        n_objects = arr.shape[0]

        # Initialize the selected index with a number outside the possible index values
        index = n_objects + 1

        # minimum similarity value that is guaranteed to be higher than all the comparisons, this
        # value should be a warranty that a exist a sample with similarity lower than min_sim. The
        # max possible similarity value for set of samples is 1.00.
        min_sim = 1.01

        # For each sample in the set, calculate the columnwise sum of the data without the sample
        comp_sums = c_total - arr

        # for each sample calculate the similarity index of the complete set without the sample
        for idx, obj in enumerate(comp_sums):
            # calculate the similarity index of the set of objects without the current object
            sim_index = self(obj, n_objects=n_objects - 1)
            # if the similarity is lower than the previous minimum similarity, update the minimum
            # similarity and the index
            if sim_index < min_sim:
                min_sim, index = sim_index, idx
            else:
                pass
        # the index of the object that increases more the similarity of the set when added is
        # returned (the medoid)
        return index

    def calculate_outlier(self, arr: np.ndarray = None, c_total=None) -> int:
        r"""Calculate the outlier of a set of real-valued vectors or binary objects.

        Calculates the outlier of a set of real-valued vectors or binary objects. Using the
        similarity index provided in the class initialization.

        Parameters
        ----------
        arr: np.array
            np.array of all the real-valued vectors or binary objects.
        c_total:
            np.array with the columnwise sums of the data, not necessary to provide.
        """
        # Check if the data is a np.ndarray of a list
        if not isinstance(arr, np.ndarray):
            raise TypeError("Input data is not a np.ndarray, please input the right data type")
        # Check if the data is one dimensional
        if arr.ndim != 2:
            raise ValueError(
                "Data must be a two dimensional np.ndarray for calculating the outlier."
            )
        # Check if the data has at least 3 rows
        if arr.shape[0] < 3:
            raise ValueError("Input data must have at least 3 rows to calculate the outlier.")

        # check if c_total is provided and if not, calculate it
        if c_total is None:
            c_total = np.sum(arr, axis=0)
        # if c_total is provided, check if it has the same number of columns as the data
        elif c_total is not None and len(arr[0]) != len(c_total):
            raise ValueError("Dimensions of objects and columnwise sum differ")

        n_objects = arr.shape[0]

        # Initialize the selected index with a number outside the possible index values
        index = n_objects + 1

        # maximum similarity value that is guaranteed to be lower than all the comparisons, this
        # value should be a warranty that a exist a sample with similarity lower than min_sim. The
        # min possible similarity value for set of samples is 0.00.
        max_sim = -0.01

        # For each sample in the set, calculate the columnwise sum of the data without the sample
        comp_sums = c_total - arr

        # for each sample calculate the similarity index of the complete set without the sample
        for idx, obj in enumerate(comp_sums):
            # calculate the similarity index of the set of objects without the current object
            sim_index = self(arr=obj, n_objects=n_objects - 1)
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


def tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute Tanimoto coefficient or index (a.k.a. Jaccard similarity coefficient).

    For two binary or non-binary arrays :math:`A` and :math:`B`, Tanimoto coefficient
    is defined as the size of their intersection divided by the size of their union:

    ..math::
        T(A, B) = \frac{| A \cap B|}{| A \cup B |} =
        \frac{| A \cap B|}{|A| + |B| - | A \cap B|} =
        \frac{A \cdot B}{\|A\|^2 + \|B\|^2 - A \cdot B}

    where :math:`A \cdot B = \sum_i{A_i B_i}` and :math:`\|A\|^2 = \sum_i{A_i^2}`.

    Parameters
    ----------
    a : ndarray of shape (n_features,)
        The 1D feature array of sample :math:`A` in an `n_features` dimensional space.
    b : ndarray of shape (n_features,)
        The 1D feature array of sample :math:`B` in an `n_features` dimensional space.

    Returns
    -------
    coeff : float
        Tanimoto coefficient between feature arrays :math:`A` and :math:`B`.

    Bajusz, D., Rácz, A., and Héberger, K.. (2015)
    Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations?.
    Journal of Cheminformatics 7.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Arguments a and b should be 1D arrays, got {a.ndim} and {b.ndim}")
    if a.shape != b.shape:
        raise ValueError(
            f"Arguments a and b should have the same shape, got {a.shape} != {b.shape}"
        )
    coeff = sum(a * b) / (sum(a**2) + sum(b**2) - sum(a * b))
    return coeff


def modified_tanimoto(a: np.array, b: np.array) -> float:
    r"""Compute the modified tanimoto coefficient from bitstring vectors of data points A and B.

    Adjusts calculation of the Tanimoto coefficient to counter its natural bias towards
    shorter vectors using a Bernoulli probability model.

    ..math::
    MT = \frac{2-p}{3}T_1 + \frac{1+p}{3}T_0

    where :math:`p` is success probability of independent trials,
    :math:`T_1` is the number of common '1' bits between data points
    (:math:`T_1 = | A \cap B |`), and :math:`T_0` is the number of common '0'
    bits between data points (:math:`T_0 = |(1-A) \cap (1-B)|`).


    Parameters
    ----------
    a : ndarray of shape (n_features,)
        The 1D bitstring feature array of sample :math:`A` in an `n_features` dimensional space.
    b : ndarray of shape (n_features,)
        The 1D bitstring feature array of sample :math:`B` in an `n_features` dimensional space.

    Returns
    -------
    mt : float
        Modified tanimoto coefficient between bitstring feature arrays :math:`A` and :math:`B`.

    Notes
    -----
    The equation above has been derived from

    ..math::
    MT_\alpha= {\alpha}T_1 + (1-\alpha)T_0

    where :math:`\alpha = \frac{2-p}{3}`. This is done so that the expected value
    of the modified tanimoto, :math:`E(MT)`, remains constant even as the number of
    trials :math:`p` grows larger.

    Fligner, M. A., Verducci, J. S., and Blower, P. E.. (2002)
    A Modification of the Jaccard-Tanimoto Similarity Index for
    Diverse Selection of Chemical Compounds Using Binary Strings.
    Technometrics 44, 110-119.
    """
    if a.ndim != 1:
        raise ValueError(f"Argument `a` should have dimension 1 rather than {a.ndim}.")
    if b.ndim != 1:
        raise ValueError(f"Argument `b` should have dimension 1 rather than {b.ndim}.")
    if a.shape != b.shape:
        raise ValueError(
            f"Arguments a and b should have the same shape, got {a.shape} != {b.shape}"
        )

    n_features = len(a)
    # number of common '1' bits between points A and B
    n_11 = sum(a * b)
    # number of common '0' bits between points A and B
    n_00 = sum((1 - a) * (1 - b))

    # calculate Tanimoto coefficient based on '0' bits
    t_1 = 1
    if n_00 != n_features:
        # bit strings are not all '0's
        t_1 = n_11 / (n_features - n_00)
    # calculate Tanimoto coefficient based on '1' bits
    t_0 = 1
    if n_11 != n_features:
        # bit strings are not all '1's
        t_0 = n_00 / (n_features - n_11)

    # combine into modified tanimoto using Bernoulli Model
    # p = independent success trials
    #       evaluated as total number of '1' bits
    #       divided by 2x the fingerprint length
    p = (n_features - n_00 + n_11) / (2 * n_features)
    # mt = x * T_1 + (1-x) * T_0
    #       x = (2-p)/3 so that E(mt) = 1/3, no matter the value of p
    mt = (((2 - p) / 3) * t_1) + (((1 + p) / 3) * t_0)
    return mt


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
