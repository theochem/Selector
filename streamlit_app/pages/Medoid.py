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

import scipy
from selector.methods.partition import Medoid

# Add the streamlit_app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from utils import *

# Set page configuration
st.set_page_config(
    page_title = "Medoid",
    page_icon = os.path.join(parent_dir, "assets" , "QC-Devs.png"),
)

st.title("Medoid Method")


description = """
    Points are initially used to construct a KDTree. Eucleidean distances are used for this
    algorithm. The first point selected is based on the starting_idx provided and becomes the first
    query point. An approximation of the furthest point to the query point is found using
    find_furthest_neighbor and is selected. find_nearest_neighbor is then done to eliminate close
    neighbors to the new selected point. Medoid is then calculated from previously selected points
    and is used as the new query point for find_furthest_neighbor, repeating the process. Terminates
    upon selecting requested number of points or if all available points exhausted.
    """

references = "Adapted from: https://en.wikipedia.org/wiki/K-d_tree#Construction"

display_sidebar_info("Medoid Method", description, references)

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

    # Parameters for Medoid
    st.info("The parameters below are optional. If not specified, default values will be used.")

    start_id = st.number_input("Index for the first point to be selected. (start_id)", value = 0, step = 1, on_change=clear_results)

    scaling = st.number_input("Percent of average maximum distance to use when eliminating the closest points. (scaling)",
                              value=10.0, step=1.0, on_change=clear_results)

    if st.button("Run Medoid Algorithm"):
        selector = Medoid(start_id=start_id, func_distance = lambda x, y: scipy.spatial.minkowski_distance(x, y) ** 2, scaling=scaling)
        selected_ids = run_algorithm(selector, matrix, num_points, labels)
        st.session_state['selected_ids'] = selected_ids

# Check if the selected indices are stored in the session state
if 'selected_ids' in st.session_state and matrix_file is not None:
    selected_ids = st.session_state['selected_ids']
    st.write("Selected indices:", selected_ids)

    export_results(selected_ids)
