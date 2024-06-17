import streamlit as st
import numpy as np
import pandas as pd
import json
import os

from sklearn.metrics import pairwise_distances
from selector.methods.distance import MaxMin

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the assets directory
assets_dir = os.path.join(current_dir, "..", "assets")

# Set page configuration
st.set_page_config(
    page_title="MaxMin",
    page_icon=os.path.join(assets_dir, "QC-Devs.png")
)

st.title("Brute Strength - MaxMin")
st.sidebar.header("Brute Strength - MaxMin")
st.sidebar.markdown(
    """
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

    References
    ----------
    [1] Ashton, Mark, et al., Identification of diverse database subsets using
    property‐based and fragment‐based molecular descriptions, Quantitative
    Structure‐Activity Relationships 21.6 (2002): 598-604.
    """
)

# File uploader for feature matrix or distance matrix (required)
matrix_file = st.file_uploader("Upload a feature matrix or distance matrix (required)", type=["csv", "xlsx", "npz", "npy"], key="matrix_file")

# Clear selected indices if a new matrix file is uploaded
if matrix_file is None:
    st.session_state.pop("selected_ids", None)

# Load data from matrix file
if matrix_file is not None:
    try:
        header_option = None
        if matrix_file.name.endswith(".csv") or matrix_file.name.endswith(".xlsx"):
            header_option = st.checkbox("Does the file have a header?", key = "header_option")
            st.warning("⚠️ Warning: This will affect the final output if not specified correctly.")

        if matrix_file.name.endswith(".csv") or matrix_file.name.endswith(".xlsx"):
            if header_option:
                # Load the matrix with header
                matrix = pd.read_csv(matrix_file).values
            else:
                # Load the matrix without header
                matrix = pd.read_csv(matrix_file, header = None).values
            st.write("Matrix shape:", matrix.shape)
            st.write(matrix)

        elif matrix_file.name.endswith(".npz"):
            matrix_data = np.load(matrix_file)
            # Select the array in the .npz file
            array_names = matrix_data.files
            selected_array = st.selectbox("Select the array to use", array_names)
            matrix = matrix_data[selected_array]
            st.write("Matrix shape:", matrix.shape)
            st.write(matrix)
        elif matrix_file.name.endswith(".npy"):
            matrix = np.load(matrix_file)
            st.write("Matrix shape:", matrix.shape)
            st.write(matrix)
    except Exception as e:
        st.error(f'An error occurred while loading matrix file: {e}')
        matrix = None


    # Input for number of points to select (required)
    num_points = st.number_input("Number of points to select", min_value=1, step=1, key="num_points")

    # Input for cluster label list (optional)
    label_file = st.file_uploader("Upload a cluster label list (optional)", type=["csv", "xlsx"], key="label_file")
    labels = None
    if label_file is not None:
        try:
            label_header_option = None
            if label_file.name.endswith(".csv") or label_file.name.endswith(".xlsx"):
                label_header_option = st.checkbox("Does the file have a header?",
                                                  key = "label_header_option")
                st.warning(
                    "⚠️ Warning: This will affect the final output if not specified correctly.")

            if label_file.name.endswith(".csv") or label_file.name.endswith(".xlsx"):
                if label_header_option:
                    labels = pd.read_csv(label_file).values.flatten()
                else:
                    labels = pd.read_csv(label_file, header = None).values.flatten()
                st.write("Cluster labels shape:", labels.shape)
                st.write(labels)
        except Exception as e:
            st.error(f'An error occurred while loading cluster label file: {e}')
            labels = None


    if st.button("Run MaxMin Algorithm"):
        try:
            # Check if the input matrix is a feature matrix or a distance matrix
            if matrix.shape[0] == matrix.shape[1]:
                # Distance matrix
                selector = MaxMin()
                selected_ids = selector.select(matrix, size = num_points, labels = labels)
            else:
                # Feature matrix
                selector = MaxMin(lambda x: pairwise_distances(x, metric = "euclidean"))
                selected_ids = selector.select(matrix, size = num_points, labels = labels)

            # Convert selected indices to a list of integers
            selected_ids = [int(i) for i in selected_ids]

            # Save selected indices to session state
            st.session_state['selected_ids'] = selected_ids
        except ValueError as ve:
            st.error(f"An error occurred while running the MaxMin algorithm: {ve}")
        except Exception as e:
            st.error(f"An error occurred while running the MaxMin algorithm: {e}")

# Check if the selected indices are stored in the session state
if 'selected_ids' in st.session_state and matrix_file is not None:
    selected_ids = st.session_state['selected_ids']
    st.write("Selected indices:", selected_ids)

    # export format
    export_format = st.selectbox("Select export format", ["CSV", "JSON"], key="export_format")

    if export_format == "CSV":
        csv_data = pd.DataFrame(selected_ids, columns = ["Selected Indices"])
        csv = csv_data.to_csv(index = False).encode('utf-8')
        st.download_button(
            label = "Download as CSV",
            data = csv,
            file_name = 'selected_indices.csv',
            mime = 'text/csv',
        )
    elif export_format == "JSON":
        json_data = json.dumps({"Selected Indices": selected_ids})
        st.download_button(
            label = "Download as JSON",
            data = json_data,
            file_name = 'selected_indices.json',
            mime = 'application/json',
        )