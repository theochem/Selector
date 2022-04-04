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

from DiverseSelector.feature import (DescriptorGenerator,
                                     feature_reader,
                                     FingerprintGenerator,
                                     get_features,
                                     )
from DiverseSelector.test.common import load_testing_mols
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import pytest

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


def test_feature_desc_invalid():
    """Testing invalid descriptor with 3D molecules."""
    # load molecules
    mols = load_testing_mols(mol_type="3d")
    # generate molecular descriptors with the DescriptorGenerator
    with pytest.raises(ValueError):
        desc_generator = DescriptorGenerator(mols=mols,
                                             desc_type="invalid_desc",
                                             use_fragment=True,
                                             ipc_avg=True,
                                             )

        desc_generator.compute_descriptor()


def test_feature_fp_secfp6():
    """Testing SECFP6 fingerprints with 3D molecules."""
    # load molecules
    mols = load_testing_mols(mol_type="3d")
    # generate molecular descriptors with the DescriptorGenerator
    fp_generator = FingerprintGenerator(mols=mols,
                                        fp_type="SECFP",
                                        n_bits=1024,
                                        radius=3,
                                        min_radius=1,
                                        random_seed=42,
                                        rings=True,
                                        isomeric=True,
                                        kekulize=False,
                                        )
    df_secfp6 = fp_generator.compute_fingerprint()
    # load the expected descriptor dataframe
    with path("DiverseSelector.test.data", "drug_mols_secfp6.csv") as secfp6_csv:
        df_secfp6_exp = pd.read_csv(secfp6_csv,
                                    sep=",",
                                    index_col=0)
    # check if the dataframes are equal
    assert_equal(df_secfp6.shape, df_secfp6_exp.shape)
    assert_almost_equal(df_secfp6.to_numpy(int),
                        df_secfp6_exp.to_numpy(int),
                        )


def test_feature_fp_ecfp6():
    """Testing ECFP6 fingerprints with 3D molecules."""
    # load molecules
    mols = load_testing_mols(mol_type="3d")
    # generate molecular descriptors with the DescriptorGenerator
    fp_generator = FingerprintGenerator(mols=mols,
                                        fp_type="ECFP",
                                        n_bits=1024,
                                        radius=3,
                                        min_radius=1,
                                        random_seed=42,
                                        rings=True,
                                        isomeric=True,
                                        kekulize=False,
                                        )
    df_ecfp6 = fp_generator.compute_fingerprint()
    # load the expected descriptor dataframe
    with path("DiverseSelector.test.data", "drug_mols_ecfp6.csv") as ecfp6_csv:
        df_ecfp6_exp = pd.read_csv(ecfp6_csv,
                                   sep=",",
                                   index_col=0)
    # check if the dataframes are equal
    assert_equal(df_ecfp6.shape, df_ecfp6_exp.shape)
    assert_almost_equal(df_ecfp6.to_numpy(int),
                        df_ecfp6_exp.to_numpy(int),
                        )


def test_feature_fp_morgan():
    """Testing Morgan fingerprints with 3D molecules."""
    # load molecules
    mols = load_testing_mols(mol_type="3d")
    # generate molecular descriptors with the DescriptorGenerator
    fp_generator = FingerprintGenerator(mols=mols,
                                        fp_type="Morgan",
                                        n_bits=1024,
                                        radius=3,
                                        min_radius=1,
                                        random_seed=42,
                                        rings=True,
                                        isomeric=True,
                                        kekulize=False,
                                        )
    df_morgan = fp_generator.compute_fingerprint()
    # load the expected descriptor dataframe
    with path("DiverseSelector.test.data", "drug_mols_morgan.csv") as morgan_csv:
        df_morgan_exp = pd.read_csv(morgan_csv,
                                    sep=",",
                                    index_col=0)
    # check if the dataframes are equal
    assert_equal(df_morgan.shape, df_morgan_exp.shape)
    assert_almost_equal(df_morgan.to_numpy(int),
                        df_morgan_exp.to_numpy(int),
                        )


def test_feature_fp_rdkit():
    """Testing Morgan fingerprints with 3D molecules."""
    # load molecules
    mols = load_testing_mols(mol_type="3d")
    # generate molecular descriptors with the DescriptorGenerator
    fp_generator = FingerprintGenerator(mols=mols,
                                        fp_type="RDkFingerprint",
                                        n_bits=1024,
                                        radius=3,
                                        min_radius=1,
                                        random_seed=42,
                                        rings=True,
                                        isomeric=True,
                                        kekulize=False,
                                        )
    df_rdkit_fp = fp_generator.compute_fingerprint()
    # load the expected descriptor dataframe
    with path("DiverseSelector.test.data", "drug_mols_RDKitfp.csv") as rdkit_fp_csv:
        df_rdkit_fp_exp = pd.read_csv(rdkit_fp_csv,
                                      sep=",",
                                      index_col=0)
    # check if the dataframes are equal
    assert_equal(df_rdkit_fp.shape, df_rdkit_fp_exp.shape)
    assert_almost_equal(df_rdkit_fp.to_numpy(int),
                        df_rdkit_fp_exp.to_numpy(int),
                        )


def test_feature_fp_maccskeys():
    """Testing MaCCSKeys fingerprints with 3D molecules."""
    # load molecules
    mols = load_testing_mols(mol_type="3d")
    # generate molecular descriptors with the DescriptorGenerator
    fp_generator = FingerprintGenerator(mols=mols,
                                        fp_type="MaCCSKeys",
                                        n_bits=1024,
                                        radius=3,
                                        min_radius=1,
                                        random_seed=42,
                                        rings=True,
                                        isomeric=True,
                                        kekulize=False,
                                        )
    df_maccskeys_fp = fp_generator.compute_fingerprint()
    # load the expected descriptor dataframe
    with path("DiverseSelector.test.data", "drug_mols_MaCCSKeys.csv") as maccskeys_fp_csv:
        df_maccskeys_fp_exp = pd.read_csv(maccskeys_fp_csv,
                                          sep=",",
                                          index_col=0)
    # check if the dataframes are equal
    assert_equal(df_maccskeys_fp.shape, df_maccskeys_fp_exp.shape)
    assert_almost_equal(df_maccskeys_fp.to_numpy(int),
                        df_maccskeys_fp_exp.to_numpy(int),
                        )


def test_feature_fp_invalid():
    """Testing invalid fingerprints with 3D molecules."""
    # load molecules
    mols = load_testing_mols(mol_type="3d")
    # generate molecular descriptors with the DescriptorGenerator
    with pytest.raises(ValueError):
        fp_generator = FingerprintGenerator(mols=mols,
                                            fp_type="invalid_fp",
                                            n_bits=1024,
                                            radius=3,
                                            min_radius=1,
                                            random_seed=42,
                                            rings=True,
                                            isomeric=True,
                                            kekulize=False,
                                            )

        fp_generator.compute_fingerprint()


def test_feature_reader_csv():
    """Testing the feature reader function."""
    # load mock features
    with path("DiverseSelector.test.data", "mock_features.csv") as mock_feature_csv:
        df_features = feature_reader(str(mock_feature_csv),
                                     sep=",",
                                     engine="python",
                                     )

    data = {"feature_1": [1, 2, 3, 4],
            "feature_2": [2, 3, 4, 5],
            "feature_3": [3, 4, 5, 6],
            "feature_4": [4, 5, 6, 7],
            "feature_5": [5, 6, 7, 8],
            }
    df_features_exp = pd.DataFrame(data)

    # check if the dataframes are equal
    assert_equal(df_features.shape, df_features_exp.shape)
    assert_almost_equal(df_features.to_numpy(int),
                        df_features_exp.to_numpy(int),
                        )


def test_feature_reader_xlsx():
    """Testing the feature reader function."""
    # load mock features
    with path("DiverseSelector.test.data", "mock_features.xlsx") as mock_feature_xlsx:
        df_features = feature_reader(str(mock_feature_xlsx),
                                     engine="openpyxl",
                                     )

    data = {"feature_1": [1, 2, 3, 4],
            "feature_2": [2, 3, 4, 5],
            "feature_3": [3, 4, 5, 6],
            "feature_4": [4, 5, 6, 7],
            "feature_5": [5, 6, 7, 8],
            }
    df_features_exp = pd.DataFrame(data)

    # check if the dataframes are equal
    assert_equal(df_features.shape, df_features_exp.shape)
    assert_almost_equal(df_features.to_numpy(int),
                        df_features_exp.to_numpy(int),
                        )


def test_feature_get_features_load():
    """Testing the feature getter function by loading features."""
    with path("DiverseSelector.test.data", "mock_features.xlsx") as mock_feature_xlsx:
        df_features = get_features(engine="openpyxl",
                                   feature_type=None,
                                   mol_file=None,
                                   feature_file=str(mock_feature_xlsx),
                                   )
    data = {"feature_1": [1, 2, 3, 4],
            "feature_2": [2, 3, 4, 5],
            "feature_3": [3, 4, 5, 6],
            "feature_4": [4, 5, 6, 7],
            "feature_5": [5, 6, 7, 8],
            }
    df_features_exp = pd.DataFrame(data)

    # check if the dataframes are equal
    assert_equal(df_features.shape, df_features_exp.shape)
    assert_almost_equal(df_features.to_numpy(int),
                        df_features_exp.to_numpy(int),
                        )


def test_feature_get_features_fp_generate():
    """Testing the feature getter function by computing features."""
    with path("DiverseSelector.test.data", "drug_mols.sdf") as sdf_drugs:
        df_secfp6 = get_features(feature_type="fingerprint",
                                 fp_type="SECFP",
                                 mol_file=str(sdf_drugs),
                                 feature_file=None,
                                 n_bits=1024,
                                 radius=3,
                                 min_radius=1,
                                 random_seed=42,
                                 rings=True,
                                 isomeric=True,
                                 kekulize=False,
                                 )
    with path("DiverseSelector.test.data", "drug_mols_secfp6.csv") as secfp6_csv:
        df_secfp6_exp = pd.read_csv(secfp6_csv,
                                    sep=",",
                                    index_col=0)

    # check if the dataframes are equal
    assert_equal(df_secfp6.shape, df_secfp6_exp.shape)
    assert_almost_equal(df_secfp6.to_numpy(int),
                        df_secfp6_exp.to_numpy(int),
                        )


def test_feature_get_features_desc_generate():
    """Testing the feature getter function by computing features."""
    with path("DiverseSelector.test.data", "drug_mols.sdf") as sdf_drugs:
        df_desc_rdkit_frag = get_features(mol_file=str(sdf_drugs),
                                          feature_type="descriptor",
                                          desc_type="rdkit_frag",
                                          use_fragment=True,
                                          ipc_avg=True,
                                          )
    with path("DiverseSelector.test.data", "drug_mols_desc_rdkit_frag.csv") as desc_csv:
        df_desc_rdkit_frag_exp = pd.read_csv(desc_csv,
                                             sep=",",
                                             )

    # check if the dataframes are equal
    assert_equal(df_desc_rdkit_frag.shape, df_desc_rdkit_frag_exp.shape)
    assert_almost_equal(df_desc_rdkit_frag.to_numpy(float),
                        df_desc_rdkit_frag_exp.to_numpy(float),
                        )
