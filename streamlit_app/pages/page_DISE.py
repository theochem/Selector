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

from utils import *

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

# File uploader for feature matrix or distance matrix (required)
matrix_file = st.file_uploader("Upload a feature matrix or distance matrix (required)",
                               type=["csv", "xlsx", "npz", "npy"], key="matrix_file", on_change=clear_results)

# Clear selected indices if a new matrix file is uploaded
if matrix_file is None:
    clear_results()
# Load data from matrix file
else:
    matrix = load_matrix(matrix_file)
    num_points = st.number_input("Number of points to select", min_value = 1, step = 1,
                                 key = "num_points", on_change=clear_results)
    label_file = st.file_uploader("Upload a cluster label list (optional)", type = ["csv", "xlsx"],
                                  key = "label_file", on_change=clear_results)
    labels = load_labels(label_file) if label_file else None

    # Parameters for Directed Sphere Exclusion
    st.info("The parameters below are optional. If not specified, default values will be used.")

    r0 = st.number_input("Initial guess for radius of exclusion sphere (r0)", value=None, step=0.1,
                         on_change=clear_results)
    ref_index = st.number_input("Reference index (ref_index)", value=0, step=1, on_change=clear_results)
    tol = st.number_input("Percentage tolerance of sample size error (tol)", value=0.05, step=0.05,
                          on_change=clear_results)
    n_iter = st.number_input("Number of iterations for optimizing the radius of exclusion sphere (n_iter)",
                             value=10, step=10, on_change=clear_results)
    p = st.number_input("Minkowski p-norm distance (p)", value=2.0, step=1.0, on_change=clear_results)
    eps = st.number_input("Approximate nearest neighbor search parameter (eps)", value=0.0, step=0.1,
                          on_change=clear_results)

    if st.button("Run DISE Algorithm"):
        selector = DISE(r0=r0, ref_index=ref_index, tol=tol, n_iter=n_iter, p=p, eps=eps)
        selected_ids = run_algorithm(selector, matrix, num_points, labels)
        st.session_state['selector'] = selector
        st.session_state['selected_ids'] = selected_ids

# Check if the selected indices are stored in the session state
if 'selected_ids' in st.session_state and matrix_file is not None:
    selected_ids = st.session_state['selected_ids']
    st.write("Selected indices:", selected_ids)

    if 'selector' in st.session_state:
        st.write("Radius of the exclusion sphere:", st.session_state['selector'].r)

    export_results(selected_ids)
