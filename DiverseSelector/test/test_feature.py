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

import os

import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal
from DiverseSelector.feature import (DescriptorGenerator,
                                     FingerprintGenerator,
                                     feature_filtering,
                                     get_features)
from rdkit import Chem


def test_feature_desc_mordred_smiles():
    """Testing molecular mordred descriptor with SMILES strings."""

    # load molecules
    mols = [Chem.MolFromSmiles(smiles) for smiles in
            ["OC(=O)[C@@H](N)Cc1[nH]cnc1",
             "OC(=O)C(=O)C",
             "CC(=O)OC1=CC=CC=C1C(=O)O"]
            ]
    # generate molecular descriptors with the DescriptorGenerator
    desc_generator = DescriptorGenerator(mols=mols,
                                         desc_type="mordred",
                                         use_fragment=True,
                                         ipc_avg=True,
                                         )
    df_mordred_desc = desc_generator.compute_descriptor(ignore_3D=True)
    # load the expected descriptor dataframe
    df_mordred_desc_exp = pd.read_csv(os.path.join("data", "drug_mols_desc_smi.csv"),
                                      sep=",",
                                      )
    df_mordred_desc_exp.drop(columns=["name"], inplace=True)
    # check if the dataframes are equal
    # assert_frame_equal(df_mordred_desc, df_mordred_desc_exp)
    assert_almost_equal(df_mordred_desc.to_numpy(float),
                        df_mordred_desc_exp.to_numpy(float),
                        decimal=7)
