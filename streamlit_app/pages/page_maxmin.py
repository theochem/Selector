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

from sklearn.metrics import pairwise_distances
from selector.methods.distance import MaxMin

# Add the streamlit_app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from utils import *

st.title("Brute Strength - MaxMin")

description = """
    MaxMin is possibly the most widely used method for dissimilarity-based
    compound selection. When presented with a dataset of samples, the
    initial point is chosen as the dataset's medoid center. Next, the second
    point is chosen to be that which is furthest from this initial point.
    Subsequently, all following points are selected via the following
    logic:

    1. Find the minimum distance from every point to the already-selected ones.
    2. Select the point which has the maximum distance among those calculated
       in the previous step.

    In the current implementation, this method requires or computes the full pairwise-distance
    matrix, so it is not recommended for large datasets.
    """


references = "[1] Ashton, Mark, et al., Identification of diverse database subsets using "\
             "property‐based and fragment‐based molecular descriptions, "\
             "Quantitative Structure‐Activity Relationships 21.6 (2002): 598-604."

display_sidebar_info("Brute Strength - MaxMin", description, references)

# File uploader for feature matrix or distance matrix (required)
matrix_file = st.file_uploader("Upload a feature matrix or distance matrix (required)",
                               type=["csv", "xlsx", "npz", "npy"], key="matrix_file", on_change=clear_results)

# Clear selected indices if a new matrix file is uploaded
if matrix_file is None:
    st.session_state.pop("selected_ids", None)
# Load data from matrix file
else:
    matrix = load_matrix(matrix_file)
    num_points = st.number_input("Number of points to select", min_value = 1, step = 1,
                                 key = "num_points", on_change=clear_results)
    label_file = st.file_uploader("Upload a cluster label list (optional)", type = ["csv", "xlsx"],
                                  key = "label_file", on_change=clear_results)
    labels = load_labels(label_file) if label_file else None

    distance_metric = st.selectbox("Select distance metric (optional)",
                                   [None, "euclidean", "manhattan", "cosine"],
                                   key = "distance_metric", on_change=clear_results)

    if distance_metric:
        fun_dist = lambda x: pairwise_distances(x, metric = distance_metric)
    else:
        fun_dist = None

    if st.button("Run MaxMin Algorithm"):
        if fun_dist:
            selector = MaxMin(fun_dist)
            selected_ids = run_algorithm(selector, matrix, num_points, labels)
        else:
            selector = MaxMin()
            selected_ids = run_algorithm(selector, matrix, num_points, labels)

        st.session_state['selector'] = selector
        st.session_state['selected_ids'] = selected_ids


# Check if the selected indices are stored in the session state
if 'selected_ids' in st.session_state and matrix_file is not None:
    selected_ids = st.session_state['selected_ids']
    st.write("Selected indices:", selected_ids)

    export_results(selected_ids)
