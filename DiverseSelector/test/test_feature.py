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

"""Testing for feature generation module."""

import pandas as pd
from numpy.testing import assert_equal, assert_almost_equal

from DiverseSelector.feature import (DescriptorGenerator)
from DiverseSelector.test.common import load_testing_mols

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_feature_desc_mordred_2d():
    """Testing molecular mordred descriptor with SMILES strings."""
    # load molecules
    mols = load_testing_mols(mol_type="2d")
    # generate molecular descriptors with the DescriptorGenerator
    desc_generator = DescriptorGenerator(mols=mols,
                                         desc_type="mordred",
                                         use_fragment=True,
                                         ipc_avg=True,
                                         )
    df_mordred_desc = desc_generator.compute_descriptor(ignore_3D=True)
    # load the expected descriptor dataframe
    with path("DiverseSelector.test.data", "drug_mols_desc_smi.csv") as smi_csv:
        df_mordred_desc_exp = pd.read_csv(smi_csv,
                                          sep=",",
                                          )
        df_mordred_desc_exp.drop(columns=["name"], inplace=True)

    # check if the dataframes are equal
    assert_equal(df_mordred_desc.shape, df_mordred_desc_exp.shape)
    assert_almost_equal(df_mordred_desc.to_numpy(float),
                        df_mordred_desc_exp.to_numpy(float),
                        decimal=7)


def test_feature_desc_mordred_3d():
    """Testing molecular mordred descriptor with 3d SDF formats."""
    # load molecules
    mols = load_testing_mols(mol_type="3d")
    # generate molecular descriptors with the DescriptorGenerator
    desc_generator = DescriptorGenerator(mols=mols,
                                         desc_type="mordred",
                                         use_fragment=True,
                                         ipc_avg=True,
                                         )
    df_mordred_desc = desc_generator.compute_descriptor(ignore_3D=False)
    # load the expected descriptor dataframe
    with path("DiverseSelector.test.data", "drug_mols_desc_sdf_3d.csv") as sdf_csv:
        df_mordred_desc_exp = pd.read_csv(sdf_csv,
                                          sep=",",
                                          )
        df_mordred_desc_exp.drop(columns=["name"], inplace=True)
    # check if the dataframes are equal
    assert_equal(df_mordred_desc.shape, df_mordred_desc_exp.shape)
    assert_almost_equal(df_mordred_desc.to_numpy(float),
                        df_mordred_desc_exp.to_numpy(float),
                        decimal=7)


def test_feature_desc_padelpy_3d():
    """Testing molecular PaDEL descriptor with SMILES strings."""
    # generate molecular descriptors with the DescriptorGenerator
    with path("DiverseSelector.test.data", "drug_mols.sdf") as sdf_drugs:
        desc_generator = DescriptorGenerator(mol_file=sdf_drugs,
                                             desc_type="padel",
                                             use_fragment=True,
                                             ipc_avg=True,
                                             )
        df_padel_desc = desc_generator.compute_descriptor()

    # load the expected descriptor dataframe
    with path("DiverseSelector.test.data", "drug_mols_desc_padel.csv") as sdf_csv:
        df_padel_desc_exp = pd.read_csv(sdf_csv,
                                        sep=",",
                                        index_col="Name")

    # check if the dataframes are equal
    assert_equal(df_padel_desc.shape, df_padel_desc_exp.shape)
    assert_almost_equal(df_padel_desc.to_numpy(float),
                        df_padel_desc_exp.to_numpy(float),
                        decimal=7)


def test_feature_desc_rdkit():
    """Testing molecular RDKit descriptor with 3d molecules."""
    # load molecules
    mols = load_testing_mols(mol_type="3d")
    # generate molecular descriptors with the DescriptorGenerator
    desc_generator = DescriptorGenerator(mols=mols,
                                         desc_type="rdkit",
                                         use_fragment=True,
                                         ipc_avg=True,
                                         )
    df_rdkit_desc = desc_generator.compute_descriptor()
    # load the expected descriptor dataframe
    with path("DiverseSelector.test.data", "drug_mols_desc_rdkit.csv") as rdkit_csv:
        df_rdkit_desc_exp = pd.read_csv(rdkit_csv,
                                        sep=",")
    # check if the dataframes are equal
    assert_equal(df_rdkit_desc.shape, df_rdkit_desc_exp.shape)
    assert_almost_equal(df_rdkit_desc.to_numpy(float),
                        df_rdkit_desc_exp.to_numpy(float),
                        decimal=7)


def test_feature_desc_rdkit_frag():
    """Testing molecular RDKit fragment descriptor with 3D molecules."""
    # load molecules
    mols = load_testing_mols(mol_type="3d")
    # generate molecular descriptors with the DescriptorGenerator
    desc_generator = DescriptorGenerator(mols=mols,
                                         desc_type="rdkit_frag",
                                         use_fragment=True,
                                         ipc_avg=True,
                                         )
    df_rdkit_desc = desc_generator.compute_descriptor()
    # load the expected descriptor dataframe
    with path("DiverseSelector.test.data", "drug_mols_desc_rdkit_frag.csv") as rdkit_frag_csv:
        df_rdkit_desc_exp = pd.read_csv(rdkit_frag_csv,
                                        sep=",")
    # check if the dataframes are equal
    assert_equal(df_rdkit_desc.shape, df_rdkit_desc_exp.shape)
    assert_almost_equal(df_rdkit_desc.to_numpy(float),
                        df_rdkit_desc_exp.to_numpy(float),
                        decimal=7)
