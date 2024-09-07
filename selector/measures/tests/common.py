# -*- coding: utf-8 -*-
#
# The Selector is a Python library of algorithms for selecting diverse
# subsets of data for machine-learning.
#
# Copyright (C) 2022-2024 The QC-Devs Community
#
# This file is part of Selector.
#
# Selector is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Selector is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Common functions for test module."""

import numpy as np

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


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
    coeff = (sum(a * b)) / (((sum(a**2)) + (sum(b**2))) ** 0.5)
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
    coeff = (2 * (sum(a * b))) / ((sum(a**2)) + (sum(b**2)))
    return coeff
