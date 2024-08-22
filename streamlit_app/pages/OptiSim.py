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

from selector.methods.distance import OptiSim

# Add the streamlit_app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from utils import *

# Set page configuration
st.set_page_config(
    page_title = "OptiSim",
    page_icon = os.path.join(parent_dir, "assets" , "QC-Devs.png"),
)

st.title("Adapted Optimizable K-Dissimilarity Selection (OptiSim)")


description = """
    The OptiSim algorithm selects samples from a dataset by first choosing the medoid center as the
    initial point. Next, points are randomly chosen and added to a subsample if they exist
    outside of radius r from all previously selected points (otherwise, they are discarded). Once k
    number of points have been added to the subsample, the point with the greatest minimum distance
    to the previously selected points is chosen. Then, the subsample is cleared and the process is
    repeated.
    """

references = "[1] J. Chem. Inf. Comput. Sci. 1997, 37, 6, 1181–1188. https://doi.org/10.1021/ci970282v"

display_sidebar_info("Adapted Optimizable K-Dissimilarity Selection (OptiSim)", description, references)

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

    r0 = st.number_input("Initial guess of radius for OptiSim algorithm (r0)", value=None, step=0.1,
                         on_change=clear_results)
    ref_index = st.number_input("Index for the sample to start selection from (ref_index)", value=0, step=1, on_change=clear_results)
    k = st.number_input("Amount of points to add to subsample (k)", value=10, step=1,
                          on_change=clear_results)
    tol = st.number_input("Percentage error of number of samples selected (tol)", value=0.01, step=0.01, on_change=clear_results)
    n_iter = st.number_input("Number of iterations to execute when optimizing the size of exclusion radius. (n_iter)",
                             value=10, step=1, on_change=clear_results)
    p = st.number_input("Minkowski p-norm distance (p)", value=2.0, step=1.0, on_change=clear_results)
    eps = st.number_input("Approximate nearest neighbor search parameter (eps)", value=0.0, step=0.1,
                          on_change=clear_results)
    random_seed = st.number_input("Seed for random selection of points be evaluated. (random_seed)", value=42, step=1, on_change=clear_results)

    if st.button("Run OptiSim Algorithm"):
        selector = OptiSim(r0=r0, ref_index=ref_index, k=k, tol=tol, n_iter=n_iter, eps=eps, p=p, random_seed=random_seed)
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
