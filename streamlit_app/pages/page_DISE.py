# The Selector library provides a set of tools for selecting a
# subset of the dataset and computing diversity.
#
# Copyright (C) 2023 The QC-Devs Community
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

import streamlit as st
import sys
import os

from selector.methods.distance import DISE

# Add the streamlit_app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from utils import display_sidebar_info, load_matrix, load_labels, run_algorithm, export_results

st.title("Directed Sphere Exclusion (DISE)")


description = """
    In a nutshell, this algorithm iteratively excludes any sample within a given radius from
    any already selected sample. The radius of the exclusion sphere is an adjustable parameter.
    Compared to Sphere Exclusion algorithm, the Directed Sphere Exclusion algorithm achieves a
    more evenly distributed subset selection by abandoning the random selection approach and
    instead imposing a directed selection.

    Reference sample is chosen based on the `ref_index`, which is excluded from the selected
    subset. All samples are sorted (ascending order) based on their Minkowski p-norm distance
    from the reference sample. Looping through sorted samples, the sample is selected if it is
    not already excluded. If selected, all its neighboring samples within a sphere of radius r
    (i.e., exclusion sphere) are excluded from being selected. When the selected number of points
    is greater than specified subset `size`, the selection process terminates. The `r0` is used
    as the initial radius of exclusion sphere, however, it is optimized to select the desired
    number of samples.
    """

references = "Gobbi, A., and Lee, M.-L. (2002). DISE: directed sphere exclusion."\
             "Journal of Chemical Information and Computer Sciences,"\
             "43(1), 317–323. https://doi.org/10.1021/ci025554v"

display_sidebar_info("Directed Sphere Exclusion (DISE)", description, references)
