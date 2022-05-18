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

import sklearn.metrics.pairwise

import DiverseSelector.distance
from DiverseSelector.utils import ExplicitBitVector, mol_loader, PandasDataFrame, RDKitMol
from mordred import Calculator, descriptors
import numpy as np
from padelpy import padeldescriptor
import pandas as pd
# from padelpy import from_sdf
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMHFPFingerprint
from sklearn.preprocessing import StandardScaler


__all__ = [
    "DescriptorGenerator",
    "FingerprintGenerator",
    "feature_filtering",
    "feature_reader",
    "compute_features",
]

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cwd, "padelpy"))


class DescriptorGenerator:
    """Compute molecular features."""

    def __init__(self,
                 mols: list,
                 ):
        self.mols = mols

    def mordred_desc(self,
                     ignore_3D: bool = False,
                     ) -> PandasDataFrame:
        """Mordred molecular descriptor generation.

        Parameters
        ----------
        ignore_3D : bool, optional
            Ignore 3D coordinates. The default=False.

        Returns
        -------
        df_features: PandasDataFrame
            A `pandas.DataFrame` object with compute Mordred descriptors.

        """
        # if only compute 2D descriptors, set ignore_3D=True
        calc = Calculator(descs=descriptors, ignore_3D=ignore_3D)
        df_features = pd.DataFrame(calc.pandas(self.mols))

        return df_features

    def padelpy_desc(self,
                     mol_file,
                     keep_csv: bool = False,
                     maxruntime: int = -1,
                     waitingjobs: int = -1,
                     threads: int = -1,
                     d_2d: bool = False,
                     d_3d: bool = False,
                     config: str = None,
                     convert3d: bool = False,
                     descriptortypes: str = None,
                     detectaromaticity: bool = False,
                     d_file: str = None,
                     fingerprints: bool = False,
                     log: bool = False,
                     maxcpdperfile: int = 0,
                     removesalt: bool = False,
                     retain3d: bool = False,
                     retainorder: bool = False,
                     standardizenitro: bool = False,
                     standardizetautomers: bool = False,
                     tautomerlist: str = None,
                     usefilenameasmolname: bool = False,
                     sp_timeout: int = None,
                     headless: bool = True) -> PandasDataFrame:
        """PADEL molecular descriptor generation.

        Parameters
        ----------
        mol_file : str
            Molecule file name.
        keep_csv : bool, optional
            If True, the csv file is kept. Default=False.
        maxruntime : int, optional
            Additional keyword arguments.
            See https://github.com/ecrl/padelpy/blob/master/padelpy/wrapper.py.

        Returns
        -------
        df_features: PandasDataFrame
            A `pandas.DataFrame` object with compute Mordred descriptors.

        """
        # if only compute 2D descriptors,
        # ignore_3D=True

        csv_fname = str(os.path.basename(self.mol_file)).split(".", maxsplit=1)[0] + \
                    "padel_descriptors.csv"

        padeldescriptor(maxruntime=maxruntime,
                        waitingjobs=waitingjobs,
                        threads=threads,
                        d_2d=d_2d,
                        d_3d=d_3d,
                        config=config,
                        convert3d=convert3d,
                        descriptortypes=descriptortypes,
                        detectaromaticity=detectaromaticity,
                        mol_dir=mol_file,
                        d_file=d_file,
                        fingerprints=fingerprints,
                        log=log,
                        maxcpdperfile=maxcpdperfile,
                        removesalt=removesalt,
                        retain3d=retain3d,
                        retainorder=retainorder,
                        standardizenitro=standardizenitro,
                        standardizetautomers=standardizetautomers,
                        tautomerlist=tautomerlist,
                        usefilenameasmolname=usefilenameasmolname,
                        sp_timeout=sp_timeout,
                        headless=headless,
                        )

        df_features = pd.read_csv(csv_fname, sep=",", index_col="Name")

        if not keep_csv:
            os.remove(csv_fname)

        return df_features

    def rdkit_desc(self,
                   use_fragment: bool = True,
                   ipc_avg: bool = True,
                   ) -> PandasDataFrame:
        # noqa: D403
        """RDKit molecular descriptor generation.

        Parameters
        ----------
        use_fragment : bool, optional
            If True, the return value includes the fragment binary descriptors like "fr_XXX".
        ipc_avg : bool, optional
            If True, the IPC descriptor calculates with avg=True option.

        Returns
        -------
        df_features: PandasDataFrame
            A `pandas.DataFrame` object with compute Mordred descriptors.

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

        arr_features = [_rdkit_descriptors_low(mol, desc_list=desc_list, ipc_avg=ipc_avg)
                        for mol in self.mols]
        df_features = pd.DataFrame(arr_features, columns=descriptor_types)

        return df_features

    def rdkit_frag_desc(self) -> PandasDataFrame:
        # noqa: D403
        """RDKit fragment features.

        Returns
        -------
        df_features: PandasDataFrame
            A `pandas.DataFrame` object with compute Mordred descriptors.

        """
        # http://rdkit.org/docs/source/rdkit.Chem.Fragments.html
        # this implementation is taken from https://github.com/Ryan-Rhys/FlowMO/blob/
        # e221d989914f906501e1ad19cd3629d88eac1785/property_prediction/data_utils.py#L111
        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        frag_features = np.zeros((len(self.mols), len(fragments)))
        for idx, mol in enumerate(self.mols):
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
                           ) -> list:
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

    features = []
    for desc_name, function in desc_list:
        if desc_name == "Ipc" and ipc_avg:
            feature = function(mol, avg=True)
        else:
            feature = function(mol)
        features.append(feature)

    return features


# feature selection
def feature_filtering():
    """Feature selection."""
    # todo: add feature selection for binary fingerprints
    pass


class FingerprintGenerator:
    """Fingerprint generator."""

    def __init__(self,
                 mols: list,
                 ) -> None:
        """Fingerprint generator.

        Parameters
        ----------
        mols : RDKitMol
            Molecule object.
        """
        self.mols = mols

        # molecule names
        mol_names = [Chem.MolToSmiles(mol) if mol.GetPropsAsDict().get("_Name") is None
                     else mol.GetProp("_Name") for mol in mols]
        self.mol_names = mol_names

    def compute_fingerprint(self,
                            fp_type: str = "SECFP",
                            n_bits: int = 2048,
                            radius: int = 3,
                            min_radius: int = 1,
                            random_seed: int = 12345,
                            rings: bool = True,
                            isomeric: bool = True,
                            kekulize: bool = False,
                            ) -> PandasDataFrame:
        """Compute fingerprints.

        Parameters
        ----------
        fp_type : str, optional
            Supported fingerprints: SECFP, ECFP, Morgan, RDKitFingerprint and MACCSkeys.
            Default="SECFP".
        n_bits : int, optional
            Number of bits of fingerprint. Default=2048.
        radius : int, optional
            The maximum radius of the substructure that is generated at each atom. Default=3.
        min_radius : int, optional
            The minimum radius that is used to extract n-grams.
        random_seed : int, optional
            The random seed number. Default=12345.
        rings : bool, optional
            Whether the rings (SSSR) are extracted from the molecule and added to the shingling.
            Default=True.
        isomeric : bool, optional
            Whether the SMILES added to the shingling are isomeric. Default=False.
        kekulize : bool, optional
            Whether the SMILES added to the shingling are kekulized. Default=True.
        """
        if fp_type.upper() in ["SECFP", "ECFP", "MORGAN", "RDKFINGERPRINT", "MACCSKEYS"]:
            fps = [self.rdkit_fingerprint_low(mol,
                                              fp_type=fp_type,
                                              n_bits=n_bits,
                                              radius=radius,
                                              min_radius=min_radius,
                                              random_seed=random_seed,
                                              rings=rings,
                                              isomeric=isomeric,
                                              kekulize=kekulize,
                                              ) for mol in self.mols]
        # todo: add support of e3fp

        # other cases
        else:
            raise ValueError(f"{fp_type} is not an supported fingerprint type.")

        df_fps = pd.DataFrame(np.array(fps), index=self.mol_names)

        return df_fps

    @staticmethod
    def rdkit_fingerprint_low(mol: RDKitMol,
                              fp_type: str = "SECFP",
                              n_bits: int = 2048,
                              radius: int = 3,
                              min_radius: int = 1,
                              random_seed: int = 12345,
                              rings: bool = True,
                              isomeric: bool = False,
                              kekulize: bool = False,
                              ) -> ExplicitBitVector:
        """
        Generate required molecular fingerprints.

        Parameters
        ----------
        mols : RDKitMol
            Molecule object.
        fp_type : str, optional
            Supported fingerprints: SECFP, ECFP, Morgan, RDKitFingerprint and MACCSkeys.
            Default="SECFP".
        n_bits : int, optional
            Number of bits of fingerprint. Default=2048.
        radius : int, optional
            The maximum radius of the substructure that is generated at each atom. Default=3.
        min_radius : int, optional
            The minimum radius that is used to extract n-grams.
        random_seed : int, optional
            The random seed number. Default=12345.
        rings : bool, optional
            Whether the rings (SSSR) are extracted from the molecule and added to the shingling.
            Default=True.
        isomeric : bool, optional
            Whether the SMILES added to the shingling are isomeric. Default=False.
        kekulize : bool, optional
            Whether the SMILES added to the shingling are kekulized. Default=True.

        Returns
        -------
        fp : ExplicitBitVector
            The computed molecular fingerprint.

        Notes
        -----
        fingerprint types:
        1. topological fingerprints: RDKFingerprint, Tanimoto, Dice, Cosine, Sokal, Russel,
        Kulczynski, McConnaughey, and Tversky
        2. MACCS keys:
        3. Atom pairs and topological torsions
        4. Morgan fingerprints (circular fingerprints): Morgan, ECFP, FCFP

        """
        # SECFP: SMILES extended connectivity fingerprint
        # https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0321-8
        if fp_type.upper() == "SECFP":
            secfp_encoder = rdMHFPFingerprint.MHFPEncoder(random_seed)
            fp = secfp_encoder.EncodeSECFPMol(mol,
                                              radius=radius,
                                              rings=rings,
                                              isomeric=isomeric,
                                              kekulize=kekulize,
                                              min_radius=min_radius,
                                              length=n_bits,
                                              )
        # ECFP
        # https://github.com/deepchem/deepchem/blob/1a2d2e9ff097fdbf58894d1f91359fe466c65810/deepchem/utils/rdkit_utils.py#L414
        # https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
        elif fp_type.upper() == "ECFP":
            # radius=3 --> ECFP6
            fp = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, nBits=n_bits,
                                                       useChirality=isomeric, useFeatures=False)
        elif fp_type.upper() == "MORGAN":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, nBits=n_bits,
                                                       useChirality=isomeric, useFeatures=True)
        # https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.RDKFingerprint
        elif fp_type.upper() == "RDKFINGERPRINT":
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
        elif fp_type == "MaCCSKeys":
            fp = MACCSkeys.GenMACCSKeys(mol)
        else:
            # todo: add more
            # https://github.com/keiserlab/e3fp
            # https://chemfp.readthedocs.io/en/latest/fp_types.html
            # https://xenonpy.readthedocs.io/en/stable/_modules/xenonpy/descriptor/fingerprint.html
            raise NotImplementedError(f"{fp_type} is not implemented yet.")

        return fp


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

    # use `str` function to support PosixPath
    if str(file_name).lower().endswith((".csv", ".txt")):
        df = pd.read_csv(file_name, sep=sep, engine=engine, *kwargs)
    elif str(file_name).lower().endswith((".xlsx", ".xls", "xlsb", ".odf", ".ods", ".odt")):
        df = pd.read_excel(file_name, engine=engine, *kwargs)

    return df


def compute_features(mol_file: str,
                     feature_name: str = "padel",
                     sep: str = ",",
                     n_bits: int = 2048,
                     radius: int = 3,
                     min_radius: int = 1,
                     random_seed: int = 12345,
                     rings: bool = True,
                     isomeric: bool = True,
                     kekulize: bool = False,
                     use_fragment: bool = True,
                     ipc_avg: bool = True,
                     normalize_features: bool = False,
                     feature_output: str = None,
                     **kwargs,
                     ) -> PandasDataFrame:
    """Compute molecular features.

    Parameters
    ----------
    mol_file : str
        SDF file name that provides molecules. Default=None.
    feature_name : str, optional
        Name of the feature to compute where "mordred", "padel", "rdkit", "rdkit_frag" denote
        molecular descriptors and "SECFP", "ECFP", "MORGAN", "RDKFINGERPRINT", "MACCSKEYS" denote
        molecular fingerprints. It is case insensitive. Default="padel".
    sep : str, optional
        Separator use for CSV like files. Default=",".
    n_bits : int, optional
        Number of bits to use for fingerprint. Default=2048.
    radius : int, optional
        Radius of the fingerprint. Default=3.
    min_radius : int, optional
        Minimum radius of the fingerprint. Default=1.
    random_seed : int, optional
        Random seed for the random number generator. Default=12345.
    rings : bool, optional
        Whether the rings (SSSR) are extracted from the molecule and added to the shingling.
        Default=True.
    isomeric : bool, optional
        Whether the SMILES added to the shingling are isomeric. Default=False.
    kekulize : bool, optional
        Whether the SMILES added to the shingling are kekulized. Default=False.
    use_fragment : bool, optional
        Whether the fragments are used to compute the molecular features. Default=True.
    ipc_avg : bool, optional
        Whether the IPC average is used to compute the molecular features. Default=True.
    normalize_features : bool, optional
        Whether the features are normalized. Default=True.
    feature_output : str, optional
        CSV file name to save the computed features. Default=None.
    **kwargs:
        Other keyword arguments.

    """
    # load molecules
    mols = mol_loader(file_name=mol_file, remove_hydrogen=False)

    # compute descriptors
    if feature_name.lower() in ["mordred",
                                "padel",
                                "rdkit",
                                "rdkit_frag",
                                ]:
        descriptor_gen = DescriptorGenerator(mols=mols,
                                             mol_file=mol_file,
                                             desc_type=feature_name,
                                             use_fragment=use_fragment,
                                             ipc_avg=ipc_avg,
                                             )
        df_features = descriptor_gen.compute_descriptor(**kwargs)
    # compute fingerprints
    elif feature_name.upper() in ["SECFP", "ECFP", "MORGAN", "RDKFINGERPRINT", "MACCSKEYS"]:
        # todo: e3fp requires 3D coordinates
        # todo: other fingerprints need 2D only
        # change molecule 3D coordinate generation accordingly
        fp_gen = FingerprintGenerator(mols=mols,
                                      fp_type=feature_name,
                                      n_bits=n_bits,
                                      radius=radius,
                                      min_radius=min_radius,
                                      random_seed=random_seed,
                                      rings=rings,
                                      isomeric=isomeric,
                                      kekulize=kekulize,
                                      )
        df_features = fp_gen.compute_fingerprint()
    else:
        raise ValueError(f"{feature_name} is not supported.")

    # drop infinities columns
    df_features_valid = df_features[np.isfinite(df_features).all(axis=1)]
    # impute missing values with zeros
    df_features_valid.fillna(0, inplace=True)

    # normalize the features when needed
    if normalize_features:
        df_features_valid.iloc[:, :] = StandardScaler().fit_transform(df_features_valid)

    # save features to output file
    if feature_output is None:
        feature_output = str(os.path.basename(mol_file)).split(".", maxsplit=1)[0] + "_features.csv"
    df_features_valid.to_csv(feature_output, sep=sep, index=False)

    return df_features_valid


func_dist = lambda x: sklearn.metrics.pairwise_distance(x, metric='euclidian')
func_dist = DiverseSelector.distance.compute_distance_matrix