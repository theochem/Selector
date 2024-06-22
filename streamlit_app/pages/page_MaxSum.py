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
import os
import sys

from selector.methods.distance import MaxSum

# Add the streamlit_app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from utils import display_sidebar_info, load_matrix, load_labels, run_algorithm, export_results


st.title("Brute Strength - MaxSum")

description = """
    Whereas the goal of the MaxMin algorithm is to maximize the minimum distance between any pair
    of distinct elements in the selected subset of a dataset, the MaxSum algorithm aims to maximize
    the sum of distances between all pairs of elements in the selected subset. When presented with
    a dataset of samples, the initial point is chosen as the dataset’s medoid center. Next, 
    the second point is chosen to be that which is furthest from this initial point. 
    Subsequently, all following points are selected via the following logic:

    1. Determine the sum of distances from every point to the already-selected ones.
    2. Select the point which has the maximum sum of distances among those calculated in the previous step.

    In the current implementation, this method requires or computes the full pairwise-distance
    matrix, so it is not recommended for large datasets.
    """


references = "[1] Borodin, Allan, Hyun Chul Lee, and Yuli Ye, Max-sum diversification, "\
             "monotone submodular functions and dynamic updates, Proceedings of the 31st ACM "\
             "SIGMOD-SIGACT-SIGAI symposium on Principles of Database Systems. 2012."

display_sidebar_info("Brute Strength - MaxSum", description, references)

# File uploader for feature matrix or distance matrix (required)
matrix_file = st.file_uploader("Upload a feature matrix or distance matrix (required)",
                               type=["csv", "xlsx", "npz", "npy"], key="matrix_file")

# Clear selected indices if a new matrix file is uploaded
if matrix_file is None:
    st.session_state.pop("selected_ids", None)

# Load data from matrix file
else:
    matrix = load_matrix(matrix_file)
    num_points = st.number_input("Number of points to select", min_value = 1, step = 1,
                                 key = "num_points")
    label_file = st.file_uploader("Upload a cluster label list (optional)", type = ["csv", "xlsx"],
                                  key = "label_file")
    labels = load_labels(label_file) if label_file else None

    if st.button("Run MaxSum Algorithm"):
        selected_ids = run_algorithm(MaxSum, matrix, num_points, labels)


# Check if the selected indices are stored in the session state
if 'selected_ids' in st.session_state and matrix_file is not None:
    selected_ids = st.session_state['selected_ids']
    st.write("Selected indices:", selected_ids)

    export_results(selected_ids)
