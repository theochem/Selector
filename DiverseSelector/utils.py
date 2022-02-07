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

import pandas as pd
from rdkit import Chem

from .feature import DescriptorGenerator, FingerprintGenerator

__all__ = [
    "ExplicitBitVect",
    "RDKitMol",
    "PandasDataFrame",
    "mol_reader",
    "feature_reader",
    "get_features",
]

PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
RDKitMol = TypeVar('rdkit.Chem.rdchem.Mol')
ExplicitBitVect = TypeVar("rdkit.DataStructs.cDataStructs.ExplicitBitVect")


def mol_reader(file_name: str,
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
    if file_name.lower().endswith(".sdf"):
        suppl = Chem.SDMolSupplier(file_name, RemoveHs=remove_hydrogen, sanitize=True)
        mols = [mol for mol in suppl]
    # SDF.GZ
    elif file_name.lower().endswith(".sdf.gz"):
        file_unzipped = gzip.open(file_name)
        suppl = Chem.ForwardSDMolSupplier(file_unzipped, RemoveHs=remove_hydrogen, sanitize=True)
        mols = [mol for mol in suppl]
    # SMILES: *.smi, *.smiles, *.txt, *.csv
    elif file_name.lower().endswith((".smi", ".smiles", ".txt", ".csv")):
        mols = []
        with open(file_name, "r") as f:
            for line in f:
                mols.append(Chem.MolFromSmiles(line.strip()))

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


def feature_reader(file_name: str,
                   sep: str = ",",
                   engine: str = "python",
                   **kwargs,
                   ) -> PandasDataFrame:
    """Load molecule features/descriptors.

    Parameters
    ----------
    file_name : str
        File name that provides molecular features.
    sep : str, optional
        Separator use for CSV like files. Default=",".
    engine : str, optional
        Engine name used for reading files, where "python" supports regular expression for CSV
        formats, “xlrd” supports old-style Excel files (.xls), “openpyxl” supports newer Excel file
        formats, “odf” supports OpenDocument file formats (.odf, .ods, .odt), “pyxlsb” supports
        binary Excel files. One should note that the dependency should be installed properly to
        make it work. Default="python".
    **kwargs
        Additional keyword arguments passed to
        `pd.read_csv()<https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_
        or `pd.read_excel()<https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html>`_.

    Returns
    -------
    df : PandasDataFrame
        A `pandas.DataFrame` object with molecular features.
    """

    if file_name.lower().endswith((".csv", ".txt")):
        df = pd.read_csv(file_name, sep=sep, engine=engine, *kwargs)
    elif file_name.lower().endswith((".xlsx", ".xls", "xlsb", ".odf", ".ods", ".odt")):
        df = pd.read_excel(file_name, engine=engine, *kwargs)

    return df


def get_features(sep: str = ",",
                 engine: str = "python",
                 feature_type=None,
                 mol_file=None,
                 feature_file=None,
                 **kwargs,
                 ) -> PandasDataFrame:
    """Compute molecular features.

    Parameters
    ----------
    sep : str, optional
        Separator used for parsing feature file if text stle file provided. Default=",".
    engine : str, optional
        Separator used for parsing feature file with `pandas`. Default="python".
    feature_type : str, optional
        Feature type: "descriptor" or "fingerprint". Default=None.
    mol_file : str, optional
        Molecule file name. Default=None.
    feature_file : str, optional
        Feature file name. Default=None.
    **kwargs:
        Other keyword arguments.

    Notes
    -----
    The user needs to specify at least one of `mol_file` and `feature_file`.

    """
    # todo: can be refactored to run faster
    # case: feature is not None, mol is None
    if mol_file is None and feature_file is not None:
        df_features = feature_reader(feature_file, sep=sep, engine=engine)
    # case: feature is None, mol is not None
    elif mol_file is not None and feature_file is None:
        mols = mol_reader(file_name=mol_file)
        # compute descriptors
        if feature_type.lower() == "descriptor":
            descriptor_gen = DescriptorGenerator(mols, **kwargs)
            df_features = descriptor_gen.compute_descriptor()
        # compute fingerprints
        elif feature_type.lower() == "fingerprint":
            # todo: e3fp requires 3D coordinates
            # todo: other fingerprints need 2D only
            # change molecule 3D coordinate generation accordingly
            fp_gen = FingerprintGenerator(mols, **kwargs)
            df_features = fp_gen.compute_fingerprint()
        else:
            raise ValueError(f"{feature_type} is not supported.")

    # case: feature is not None, mol is not None
    elif mol_file is not None and feature_file is not None:
        df_features = feature_reader(feature_file)
    # case: feature is None, mol is None
    else:
        raise ValueError("It is required to define the input molecule file or feature file.")

    return df_features


def dump_mol():
    """Save molecules."""
    pass


def dump_feature():
    """Save selected molecule with features."""
    pass

# todo: dump_selected_index
# todo: add index of selected molecules in base.py
