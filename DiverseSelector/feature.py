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

"""Feature generation module."""
import os
import sys

import numpy as np
import pandas as pd
from e3fp.pipeline import fprints_from_mol
from mhfp.encoder import MHFPEncoder
from mordred import Calculator, descriptors
from padelpy import from_sdf
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, PandasTools, Descriptors
from rdkit.Chem import rdMHFPFingerprint

from .utils import PandasDataFrame, RDKitMol

__all__ = [
    "descriptor_generator",
    "fingerprint_generator",
    "feature_filtering",
]

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cwd, "padelpy"))


class DescriptorGenerator:
    def __init__(self,
                 mols: list,
                 desc_type: str,
                 use_fragment: bool = True,
                 ipc_avg: bool = True,
                 ) -> None:
        self.mols = mols
        self.desc_type = desc_type
        self.use_fragment = use_fragment
        self.ipc_avg = ipc_avg
        # self.__dict__.update(kwargs)

    def descriptor_generator(self,
                             **kwargs):
        """Molecule feature generation."""
        if self.desc_type.lower() == "mordred":
            df_features = self.mordred_descriptors(self.mols)
        elif self.desc_type.lower() == "padel":
            df_features = self.padelpy_descriptors(self.mols)
        elif self.desc_type.lower() == "rdkit":
            df_features = self.rdkit_descriptors(self.mols,
                                                 use_fragment=self.use_fragment,
                                                 ipc_avg=self.ipc_avg,
                                                 *kwargs)
        elif self.desc_type.lower() == "rdkit_frag":
            df_features = self.rdkit_fragment_descriptors(self.mols)
        else:
            raise ValueError(f"Unknown descriptor type {self.desc_type}.")

        return df_features

    @staticmethod
    def mordred_descriptors(mols: list) -> PandasDataFrame:
        """Mordred molecular descriptor generation.

        Parameters
        ----------
        mols : list
            A list of molecule RDKitMol objects.

        Returns
        -------
        df_features: PandasDataFrame
            A `pandas.DataFrame` object with compute Mordred descriptors.

        """
        # if only compute 2D descriptors,
        # ignore_3D=True
        calc = Calculator(descriptors, ignore_3D=False)
        df_features = pd.DataFrame(calc.pandas(mols))

        return df_features

    @staticmethod
    def padelpy_descriptors(mols: list) -> PandasDataFrame:
        """PADEL molecular descriptor generation.

        Parameters
        ----------
        mols : list
            A list of molecule RDKitMol objects.

        Returns
        -------
        df_features: PandasDataFrame
            A `pandas.DataFrame` object with compute Mordred descriptors.

        """
        # if only compute 2D descriptors,
        # ignore_3D=True

        # save file temporarily
        writer = Chem.SDWriter("padelpy_out_tmp.sdf")
        for mol in mols:
            writer.write(mol)
        writer.close()

        desc = from_sdf(sdf_file="padelpy_out_tmp.sdf",
                        output_csv=None,
                        descriptors=True,
                        fingerprints=False,
                        timeout=None)
        df_features = pd.DataFrame(desc)

        # delete temporary file
        os.remove("padelpy_out_tmp.sdf")

        return df_features

    @staticmethod
    def rdkit_descriptors(mols: list,
                          use_fragment: bool = True,
                          ipc_avg: bool = True,
                          **kwargs,
                          ):
        """
        Rdkit molecular descriptor generation.

        Notes
        =====
        """

        # parsing descriptor information
        desc_list = []
        descriptor_types = []
        for descriptor, function in Descriptors.descList:
            if use_fragment is False and descriptor.startswith("fr_"):
                continue
            descriptor_types.append(descriptor)
            desc_list.append((descriptor, function))

        # check initialization
        assert len(descriptor_types) == len(desc_list)

        arr_features = [_rdkit_descriptors_low(mol, desc_list=desc_list, ipc_avg=ipc_avg, *kwargs)
                        for mol in mols]
        df_features = pd.DataFrame(arr_features, columns=descriptor_types)

        return df_features

    @staticmethod
    def rdkit_fragment_descriptors(mols: list):
        """RDKit fragment features."""
        # http://rdkit.org/docs/source/rdkit.Chem.Fragments.html
        # this implementation is taken from https://github.com/Ryan-Rhys/FlowMO/blob/
        # e221d989914f906501e1ad19cd3629d88eac1785/property_prediction/data_utils.py#L111
        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        frag_features = np.zeros((len(mols), len(fragments)))
        for idx, mol in enumerate(mols):
            features = [fragments[d](mol) for d in fragments]
            frag_features[idx, :] = features

        feature_names = [desc[0] for desc in Descriptors.descList[115:]]
        df_features = pd.DataFrame(data=frag_features, columns=feature_names)

        return df_features


# this part is modified from
# https://github.com/deepchem/deepchem/blob/master/deepchem/feat/molecule_featurizers/
# rdkit_descriptors.py#L11-L98
def _rdkit_descriptors_low(mol: RDKitMol,
                           desc_list: list,
                           ipc_avg: bool = True,
                           **kwargs) -> list:
    """Calculate RDKit descriptors.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit Mol object.
    desc_list: list
        A list of tuples, which contain descriptor types and functions.
    use_fragment : bool, optional
        If True, the return value includes the fragment binary descriptors like "fr_XXX".
        Default=True.
    ipc_avg : bool, optional
        If True, the IPC descriptor calculates with avg=True option. Default=True

    Returns
    -------
    features : list
        1D list of RDKit descriptors for `mol`. The length is `len(descriptors)`.
    """
    if "mol" in kwargs:
        mol = kwargs.get("mol")
        raise DeprecationWarning(
            "Mol is being phased out as a parameter, please pass RDKit mol object instead.")

    features = []
    for desc_name, function in desc_list:
        if desc_name == "Ipc" and ipc_avg:
            feature = function(mol, avg=True)
        else:
            feature = function(mol)
        features.append(feature)
    # return np.asarray(features)
    return features


# feature selection
def feature_filtering():
    """Feature selection."""
    # todo: add feature selection for binary fingerprints
    pass


def fingerprint_generator(input_mol_fname,
                          out_excel_fname,
                          inchi_col_name=None,
                          delimiter=",",
                          # smiles_column=0,
                          # name_column=1,
                          header=0,
                          # title_line=True,
                          sanitize=True,
                          removeHs=False,
                          fingerprint_name="MHFP",
                          n_bits=2048,
                          radius=3,
                          min_radius=1,
                          random_seed=12345,
                          rings=True,
                          isomeric=True,
                          kekulize=False,
                          ):
    """Wrapper to compute molecular fingerprints."""

    # file format determination
    if input_mol_fname.lower().endswith(".sdf"):
        # suppl = Chem.SDMolSupplier(input_mol_fname, sanitize=sanitize, removeHs=False)
        df = PandasTools.LoadSDF(input_mol_fname,
                                 idName="ID",
                                 molColName="ROMol",
                                 includeFingerprints=False,
                                 isomericSmiles=isomeric,
                                 smilesName=None,
                                 embedProps=False,
                                 removeHs=removeHs,
                                 strictParsing=True)

    elif input_mol_fname.lower().endswith(".smi") or input_mol_fname.lower().endswith(".csv"):
        df = pd.read_csv(input_mol_fname, sep=delimiter, header=header)
        df["ROMol"] = df[inchi_col_name].apply(lambda x: Chem.inchi.MolFromInchi(x,
                                                                                 removeHs=removeHs,
                                                                                 sanitize=sanitize))
    elif input_mol_fname.lower().endswith(".xlsx") or input_mol_fname.lower().endswith(".xls"):
        df = pd.read_excel(input_mol_fname, header=header, engine="openpyxl")
        df["ROMol"] = df[inchi_col_name].apply(lambda x: Chem.inchi.MolFromInchi(x,
                                                                                 removeHs=removeHs,
                                                                                 sanitize=sanitize))

    else:
        raise NotImplementedError("We only support SDF, SMI, XLSX and CSV file formats only.")

    # compute molecular fingerprints
    for idx, row in df.iterrows():
        fp = rdkit_fingerprint(mol=row["ROMol"],
                               fingerprint_name=fingerprint_name,
                               n_bits=n_bits,
                               radius=radius,
                               min_radius=min_radius,
                               random_seed=random_seed,
                               rings=rings,
                               isomeric=isomeric,
                               kekulize=kekulize,
                               sanitize=sanitize,
                               )
        df.loc[idx, "fingerprint"] = fp

    # save computed fingerprint along with other records
    df.drop(["ROMol"], axis=1, inplace=True)
    df.to_excel(out_excel_fname, index=None)


def rdkit_fingerprint(mol,
                      fingerprint_name="MHFP",
                      n_bits=2048,
                      radius=3,
                      min_radius=1,
                      random_seed=12345,
                      rings=True,
                      isomeric=True,
                      kekulize=False,
                      sanitize=True,
                      ):
    """
    Generate required molecular fingerprints.

    Parameters
    ----------
    fingerprint_name : str, optional
        Fingerprint name which can be chosen from that in the Notes section.
    Notes
    -----
    fingerprint types:
    1. topological fingerprints: RDKFingerprint, Tanimoto, Dice, Cosine, Sokal, Russel,
    Kulczynski, McConnaughey, and Tversky
    2. MACCS keys:
    3. Atom pairs and topological torsions
    4. Morgan fingerprints (circular fingerprints): Morgan, ECFP, FCFP
    5. MHFP and SECFP (https://www.rdkit.org/docs/source/rdkit.Chem.rdMHFPFingerprint.html and
    https://github.com/rdkit/rdkit/issues/3526)
    """
    # radius=3 -> MHFP6
    if fingerprint_name == "MHFP":
        # https://www.rdkit.org/docs/source/rdkit.Chem.rdMHFPFingerprint.html
        # https://github.com/reymond-group/mhfp
        # n_permutations: analogous to the number of bits ECFP fingerprints are folded into.
        # Higher is better, lower is less exact. 0 denotes 2048.
        # seed: seed for the MinHash operation. Has to be the same for comparable fingerprints.
        # todo: MHFP is not binary and will result to a problem when executing fp.ToBitString()

        # mhfp_encoder = rdMHFPFingerprint.MHFPEncoder(n_bits, random_seed)
        # fp = mhfp_encoder.EncodeMol(mol,
        #                             radius=radius, rings=rings, isomeric=isomeric,
        #                             kekulize=kekulize, min_radius=min_radius)

        mhfp_encoder = MHFPEncoder(n_permutations=n_bits, seed=random_seed)
        smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=False, canonical=True)
        fp = mhfp_encoder.encode(smi,
                                 radius=radius, rings=rings,
                                 kekulize=kekulize, sanitize=sanitize)
    # SECFP: SMILES extended connectivity fingerprint
    # https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0321-8
    elif fingerprint_name == "SECFP":
        secfp_encoder = rdMHFPFingerprint.MHFPEncoder(n_bits, random_seed)
        fp = secfp_encoder.EncodeSECFPMol(mol,
                                          radius=radius, rings=rings, isomeric=isomeric,
                                          kekulize=kekulize, min_radius=min_radius)
    # ECFP
    # https://github.com/deepchem/deepchem/blob/1a2d2e9ff097fdbf58894d1f91359fe466c65810/deepchem/utils/rdkit_utils.py#L414
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
    elif fingerprint_name == "ECFP":
        # radius=3 --> ECFP6
        fp = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, nBits=n_bits,
                                                   useChirality=isomeric, useFeatures=False)
    elif fingerprint_name == "Morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, nBits=n_bits,
                                                   useChirality=isomeric, useFeatures=True)
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.RDKFingerprint
    elif fingerprint_name == "RDKFingerprint":
        fp = Chem.rdmolops.RDKFingerprint(mol=mol,
                                          minPath=1,
                                          # maxPath=mol.GetNumBonds(),
                                          maxPath=10,
                                          fpSize=n_bits,
                                          nBitsPerHash=2,
                                          useHs=True,
                                          tgtDensity=0,
                                          minSize=128,
                                          branchedPaths=True,
                                          useBondOrder=True,
                                          )
    # SMARTS-based implementation of the 166 public MACCS keys
    # https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity
    elif fingerprint_name == "MaCCSKeys":
        fp = MACCSkeys.GenMACCSKeys(mol)
    else:
        # todo: add more
        # https://github.com/keiserlab/e3fp
        # https://chemfp.readthedocs.io/en/latest/fingerprint_types.html
        # https://xenonpy.readthedocs.io/en/stable/_modules/xenonpy/descriptor/fingerprint.html
        raise NotImplementedError("{} not implemented yet.".format(fingerprint_name))

    # rdkit.DataStructs.ExplicitBitVect(fp)
    if fingerprint_name != "MHFP":
        return fp.ToBitString()
    else:
        return fp


def e3fp_fingerprint(n_bits=2048,
                     radius=3,
                     rings=True,
                     isomeric=True,
                     removeHs=False,
                     sanitize=True,
                     kekulize=True,
                     random_seed=None,
                     input_sdf="../data_source/ChembiSolvExp_aqueous_optimized.sdf",
                     input_excel="../data_source/complete_solv_energies_id_20210811_v3.xlsx",
                     output_fname="e3fp_fingerprints.xlsx"):
    """E3FP fingerprint."""
    # load molecule file
    # df = PandasTools.LoadSDF(input_sdf,
    #                          idName="ID",
    #                          molColName="ROMol",
    #                          includeFingerprints=False,
    #                          isomericSmiles=isomeric,
    #                          smilesName=None,
    #                          embedProps=False,
    #                          removeHs=removeHs,
    #                          strictParsing=True)
    suppl = Chem.SDMolSupplier(input_sdf, removeHs=removeHs, sanitize=sanitize)
    mols = [mol for mol in suppl]
    # filter out molecules with only two atoms as /e3fp/fingerprint/fprinter.py line 547,
    # requires a 2-dimensional array
    # in __init__
    #     distance_matrix = array_ops.make_distance_matrix(atom_coords)
    mols_doable = [mol for mol in mols if mol.GetNumAtoms() != 2]
    mols_not_doable = [mol for mol in mols if mol.GetNumAtoms() == 2]
    # molecular name for diatomic molecule
    mol_names_doable = [mol.GetProp("_Name") for mol in mols_doable]
    # molecular name for molecules with one atom or more than two atoms
    mol_names_two_atoms = [mol.GetProp("_Name") for mol in mols_not_doable]

    # e3fp configuration
    # https://e3fp.readthedocs.io/en/latest/usage/config.html#configuration
    # https://e3fp.readthedocs.io/en/latest/_modules/e3fp/fingerprint/generate.html
    fprint_params = {"bits": n_bits,
                     "radius_multiplier": radius,
                     "first": 1,
                     "stereo": isomeric,
                     "counts": False,
                     # Use the atom invariants used by RDKit for its Morgan fingerprint
                     "rdkit_invariants": False,
                     "level": -1,
                     "include_disconnected": False,
                     "remove_duplicate_substructs": True,
                     "exclude_floating": True,
                     "overwrite": True,
                     }

    # generate e3fp fingerprint from SDF files
    # fps = map(fprints_from_mol, mols, fprint_params)
    fps = [fprints_from_mol(mol, fprint_params=fprint_params) for mol in mols_doable]
    fps_folded = np.array([fp[0].fold().to_vector(sparse=False, dtype=int) for fp in fps])

    # save to EXCEL file format
    df = pd.DataFrame(data=fps_folded, index=mol_names_doable)
    df.to_excel(output_fname)

    return fps_folded, mol_names_doable, mol_names_two_atoms


def MHFP_fingerprint(n_bits=2048,
                     radius=3,
                     rings=True,
                     removeHs=False,
                     sanitize=True,
                     kekulize=True,
                     random_seed=None,
                     input_excel="../data_source/complete_solv_energies_id_20210811_v3.xlsx",
                     output_fname="MHFP_fingerprints.xlsx"):
    """Compute MHFP fingerprint."""
    df = pd.read_excel(input_excel)

    mhfp_encoder = MHFPEncoder(n_permutations=n_bits, seed=random_seed)
    # smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=kekulize, canonical=canonical)
    # fp = mhfp_encoder.encode(smi,
    #                      radius=radius, rings=rings,
    #                      kekulize=kekulize,  sanitize=sanitize)

    output_csv = output_fname.replace(".xlsx", ".csv")
    with open(output_csv, "w+") as f:
        for idx, row in df.iterrows():
            mol = Chem.inchi.MolFromInchi(row["inchi"],
                                          sanitize=sanitize,
                                          removeHs=removeHs,
                                          logLevel=None,
                                          treatWarningAsError=False)
            mol = Chem.AddHs(mol)
            fp_vals = ",".join(map(str, mhfp_encoder.encode_mol(mol,
                                                                radius=radius, rings=rings,
                                                                kekulize=kekulize)))
            f.write(fp_vals + "\n")

    df_mhfp = pd.read_csv(output_csv, sep=",")
    df_key_info = df[["ChembiSolvExp_id", "deltaG_chembiosolv(kcal/mol)",
                      "deltaG_chembiosolv(kcal/mol)_std", "diff_SMD_chembisolv"]]
    df_result = pd.concat([df_key_info, df_mhfp], axis=1)

    df_result.to_excel(output_fname, index=None)
