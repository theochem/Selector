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

"""Common functions for test module."""

from typing import Any, Tuple, Union

import numpy as np
from rdkit import Chem
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path

__all__ = [
    "generate_synthetic_data",
    "load_testing_mols",
]


def generate_synthetic_data(n_samples: int = 100,
                            n_features: int = 2,
                            n_clusters: int = 2,
                            cluster_std: float = 1.0,
                            center_box: Tuple[float, float] = (-10.0, 10.0),
                            metric: str = "euclidean",
                            shuffle: bool = True,
                            random_state: int = 42,
                            pairwise_dist: bool = False,
                            **kwargs: Any,
                            ) -> Union[Tuple[np.ndarray, np.ndarray],
                                       Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate synthetic data.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples. Default=100.
    n_features : int, optional
        The number of features. Default=2.
    n_clusters : int, optional
        The number of clusters. Default=2.
    cluster_std : float, optional
        The standard deviation of the clusters. Default=1.0.
    center_box : tuple[float, float], optional
        The bounding box for each cluster center when centers are generated at random.
        Default=(-10.0, 10.0).
    metric : str, optional
        The metric used for computing pairwise distances. For the supported
        distance matrix, please refer to
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html.
        Default="euclidean".
    shuffle : bool, optional
        Whether to shuffle the samples. Default=True.
    random_state : int, optional
        The random state used for generating synthetic data. Default=42.
    pairwise_dist : bool, optional
        If True, then compute and return the pairwise distances between samples. Default=False.
    **kwargs : Any, optional
            Additional keyword arguments for the scikit-learn `pairwise_distances` function.

    Returns
    -------
    syn_data : np.ndarray
        The synthetic data.
    class_labels : np.ndarray
        The integer labels for cluster membership of each sample.
    dist: np.ndarray
        The symmetric pairwise distances between samples.

    """
    # pylint: disable=W0632
    syn_data, class_labels = make_blobs(n_samples=n_samples,
                                        n_features=n_features,
                                        centers=n_clusters,
                                        cluster_std=cluster_std,
                                        center_box=center_box,
                                        shuffle=shuffle,
                                        random_state=random_state,
                                        return_centers=False,
                                        )
    if pairwise_dist:
        dist = pairwise_distances(X=syn_data,
                                  Y=None,
                                  metric=metric,
                                  **kwargs,
                                  )
        return syn_data, class_labels, dist
    else:
        return syn_data, class_labels


def load_testing_mols(mol_type: str = "2d") -> list:
    """Load testing molecules.

    Parameters
    ----------
    mol_type : str, optional
        The type of molecules, "2d" or "3d". Default="2d".

    Returns
    -------
    mols : list
        The list of RDKit molecules.
    """
    if mol_type == "2d":
        mols = [Chem.MolFromSmiles(smiles) for smiles in
                ["OC(=O)[C@@H](N)Cc1[nH]cnc1",
                 "OC(=O)C(=O)C",
                 "CC(=O)OC1=CC=CC=C1C(=O)O"]
                ]
    elif mol_type == "3d":
        with path("DiverseSelector.test.data", "drug_mols.sdf") as sdf_file:
            suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
            mols = [mol for mol in suppl]
    else:
        raise ValueError("mol_type must be either '2d' or '3d'.")

    return mols


def bit_cosine(a, b):
    """Compute dice coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features in bit string.
    b : array_like
        molecules B's features in bit string.

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B.
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for idx, _ in enumerate(a):
        if a[idx] == b[idx] and a[idx] != 0:
            c += 1
    b_c = c / ((a_feat * b_feat) ** 0.5)
    return b_c


def bit_dice(a, b):
    """Compute dice coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features.
    b : array_like
        molecules B's features.

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B.
    """
    a_feat = np.count_nonzero(a)
    b_feat = np.count_nonzero(b)
    c = 0
    for idx, _ in enumerate(a):
        if a[idx] == b[idx] and a[idx] != 0:
            c += 1
    b_d = (2 * c) / (a_feat + b_feat)
    return b_d


def cosine(a, b):
    """Compute cosine coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features.
    b : array_like
        molecules B's features.

    Returns
    -------
    coeff : int
        cosine coefficient for molecule A and B.
    """
    coeff = (sum(a * b)) / (((sum(a ** 2)) + (sum(b ** 2))) ** 0.5)
    return coeff


def dice(a, b):
    """Compute dice coefficient.

    Parameters
    ----------
    a : array_like
        molecule A's features.
    b : array_like
        molecules B's features.

    Returns
    -------
    coeff : int
        dice coefficient for molecule A and B.
    """
    coeff = (2 * (sum(a * b))) / ((sum(a ** 2)) + (sum(b ** 2)))
    return coeff
