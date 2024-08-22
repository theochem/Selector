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

from selector.methods.partition import GridPartition

# Add the streamlit_app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from utils import *

# Set page configuration
st.set_page_config(
    page_title = "GridPartition",
    page_icon = os.path.join(parent_dir, "assets" , "QC-Devs.png"),
)

st.title("Grid Partitioning Method")


description = """
    Given the number of bins along each axis, samples are partitioned using various methods:

    1. The equisized_independent partitions the feature space into bins of equal size along each dimension.

    2. The equisized_dependent partitions the space where the bins can have different length in each dimension. I.e., the `l-`th dimension bins depend on the previous dimensions. So, the order of features affects the outcome.

    3. The equifrequent_independent divides the space into bins with approximately equal number of sample points in each bin.

    4. The equifrequent_dependent is similar to equisized_dependent where the partition in each dimension will depend on the previous dimensions.
    """

references = "[1] Bayley, Martin J., and Peter Willett. “Binning schemes for partition-based compound selection.” Journal of Molecular Graphics and Modelling 17.1 (1999): 10-18."

display_sidebar_info("Grid Partitioning Method", description, references)

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

    numb_bins_axis = st.number_input("Number of bins to partition each axis into (numb_bins_axis)", value = 1, step = 1)

    # Parameters for Directed Sphere Exclusion
    st.info("The parameters below are optional. If not specified, default values will be used.")

    grid_method = st.selectbox("Method used to partition the sample points into bins. (grid_method)", ["equisized_independent",
                                "equisized_dependent", "equifrequent_independent", "equifrequent_dependent"], on_change=clear_results)

    random_seed = st.number_input("Seed for random selection of sample points from each bin. (random_seed)", value=42, step=1, on_change=clear_results)

    if st.button("Run GridPartition Algorithm"):
        selector = GridPartition(numb_bins_axis, grid_method, random_seed)
        selected_ids = run_algorithm(selector, matrix, num_points, labels)
        st.session_state['selected_ids'] = selected_ids

# Check if the selected indices are stored in the session state
if 'selected_ids' in st.session_state and matrix_file is not None:
    selected_ids = st.session_state['selected_ids']
    st.write("Selected indices:", selected_ids)

    export_results(selected_ids)
