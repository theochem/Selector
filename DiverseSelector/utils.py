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

"""Utils module."""

import gzip
from typing import TypeVar

import numpy as np
# pylint: disable=W0611
from rdkit import Chem


__all__ = [
    "ExplicitBitVect",  # noqa: F822
    "RDKitMol",
    "PandasDataFrame",
    "mol_loader",
    "distance_to_similarity",
]


sklearn_supported_metrics = [
    "cityblock",
    "cosine",
    "euclidean",
    "l1",
    "l2",
    "manhattan",
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "kulsinski",
    "mahalanobis",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]


PandasDataFrame = TypeVar("DataFrame")
RDKitMol = TypeVar("Mol")
ExplicitBitVector = TypeVar("ExplicitBitVect")


def mol_loader(
    file_name: str,
    remove_hydrogen: bool = False,
    # force_field: str = None,
) -> list:
    """Load molecules as RDKit object.

    Parameters
    ----------
    file_name : str
        Input molecular file.
    remove_hydrogen : bool, optional
        To remove the hydrogen atoms. Default=False.

    Returns
    -------
    mols: list
        A list of RDKit molecule object.
    """

    # SDF
    # use `str` function to support PosixPath
    if str(file_name).lower().endswith(".sdf"):
        suppl = Chem.SDMolSupplier(file_name, removeHs=remove_hydrogen, sanitize=True)
        mols = [mol for mol in suppl]
    # SDF.GZ
    elif str(file_name).lower().endswith(".sdf.gz"):
        file_unzipped = gzip.open(file_name)
        suppl = Chem.ForwardSDMolSupplier(
            file_unzipped, removeHs=remove_hydrogen, sanitize=True
        )
        mols = [mol for mol in suppl]
    # SMILES: *.smi, *.smiles, *.txt, *.csv
    elif str(file_name).lower().endswith((".smi", ".smiles", ".txt", ".csv")):
        mols = []
        with open(file_name, "r", encoding="utf8") as f:
            for line in f:
                mols.append(Chem.MolFromSmiles(line.strip()))
    else:
        raise ValueError("Unsupported file type.")

    # todo:
    # check if needed to add Hs
    # generate 3D and minimize it with force field

    # check if 3D or not
    # with_3d_coord = True
    # for mol in mols:
    #     try:
    #         AllChem.CalcPBF(mol)
    #     except RuntimeError:
    #         with_3d_coord = False
    #         break

    # wrap a private function which takes a molecule object as input
    # 1. check if 3D coordinates
    # 2. add Hs
    # 3. generate 3D coordinates if needed
    # then we can vectorize this function

    return mols


def pick_initial_compounds(arr_dist):
    """
    Pick the initial compounds using medioid.

    Parameters
    ----------
    arr_dist: np.ndarray

    Returns
    -------
    starting_idx: int
        center of the medioid
    """
    # use the molecule with maximum distance to initial medoid as  the starting molecule
    # https://www.sciencedirect.com/science/article/abs/pii/S1093326399000145?via%3Dihub
    # J. Mol. Graphics Mod., 1998, Vol. 16,
    # DISSIM: A program for the analysis of chemical diversity
    medoid_idx = np.argmin(arr_dist.sum(axis=0))

    # selected molecule with maximum distance to medoid
    starting_idx = np.argmax(arr_dist[medoid_idx, :])
    return starting_idx


def dump_mol():
    """Save molecules."""
    pass


def dump_feature():
    """Save selected molecule with features."""
    pass


# todo: dump_selected_index
# todo: add index of selected molecules in base.py


def distance_to_similarity(x: np.ndarray, dist: bool = True) -> np.ndarray:
    """Convert between distance and similarity matrix.

    Parameters
    ----------
    x : ndarray
        Symmetric distance or similarity array.
    dist : bool
        Confirms the matrix is distance.

    Returns
    -------
    y : ndarray
        Symmetric distance or similarity array.
    """
    if dist is True:
        y = 1 / (1 + x)
    else:
        y = (1 / x) - 1
    return y
