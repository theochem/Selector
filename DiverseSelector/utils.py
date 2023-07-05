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


import numpy as np


__all__ = [
    "ExplicitBitVector",
    "RDKitMol",
    "PandasDataFrame",
    "mol_loader",
    "distance_to_similarity",
]


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
