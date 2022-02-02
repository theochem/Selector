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

__all__ = [
    "mol_reader",
    "feature_reader",
    "feature_generator",
    "get_features",
]


def mol_reader():
    """Load molecules as RDKit object."""
    pass


def feature_reader():
    """Load molecule features/descriptors."""
    pass


def feature_generator():
    """Molecule feature generation."""
    pass


def get_features(mol_file,
                 feature_file,
                 ):
    """Compute molecular features."""
    # todo: can be refactored to run faster

    # case: feature is not None, mol is None
    if mol_file is None and feature_file is not None:
        features = feature_reader(feature_file)
    # case: feature is None, mol is not None
    elif mol_file is not None and feature_file is None:
        features = feature_generator(mol_file)
    # case: feature is not None, mol is not None
    elif mol_file is not None and feature_file is not None:
        features = feature_reader(feature_file)
    # case: feature is None, mol is None
    else:
        raise ValueError("It is required to define the input molecule file or feature file.")

    return features
