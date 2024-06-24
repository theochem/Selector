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
import numpy as np
import pandas as pd
import json
import os

from sklearn.metrics import pairwise_distances

def set_page_config(page_title, page_icon):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "..", "assets")

    st.set_page_config(
        page_title=page_title,
        page_icon=os.path.join(assets_dir, page_icon)
    )

def display_sidebar_info(title, description, references):
    st.sidebar.header(title)
    st.sidebar.info(description)
    st.sidebar.title("References")
    st.sidebar.info(references)

# Load data from matrix file
def load_matrix(matrix_file):
    try:
        header_option = None
        if matrix_file.name.endswith(".csv") or matrix_file.name.endswith(".xlsx"):
            header_option = st.checkbox("Does the file have a header?", key="header_option")
            st.warning("Warning: This will affect the final output if not specified correctly.")

        if matrix_file.name.endswith(".csv") or matrix_file.name.endswith(".xlsx"):
            if header_option:
                # Load the matrix with header
                matrix = pd.read_csv(matrix_file).values
            else:
                # Load the matrix without header
                matrix = pd.read_csv(matrix_file, header=None).values
            st.write("Matrix shape:", matrix.shape)
            st.write(matrix)
        elif matrix_file.name.endswith(".npz"):
            matrix_data = np.load(matrix_file)
            array_names = matrix_data.files # Select the array in the .npz file
            selected_array = st.selectbox("Select the array to use", array_names)
            matrix = matrix_data[selected_array]
            st.write("Matrix shape:", matrix.shape)
            st.write(matrix)
        elif matrix_file.name.endswith(".npy"):
            matrix = np.load(matrix_file)
            st.write("Matrix shape:", matrix.shape)
            st.write(matrix)
        return matrix
    except Exception as e:
        st.error(f'An error occurred while loading matrix file: {e}')
        return None

def load_labels(label_file):
    try:
        label_header_option = None
        if label_file.name.endswith(".csv") or label_file.name.endswith(".xlsx"):
            label_header_option = st.checkbox("Does the file have a header?", key="label_header_option")
            st.warning("Warning: This will affect the final output if not specified correctly.")

        if label_file.name.endswith(".csv") or label_file.name.endswith(".xlsx"):
            if label_header_option:
                labels = pd.read_csv(label_file).values.flatten()
            else:
                labels = pd.read_csv(label_file, header=None).values.flatten()
            st.write("Cluster labels shape:", labels.shape)
            st.write(labels)
        return labels
    except Exception as e:
        st.error(f'An error occurred while loading cluster label file: {e}')
        return None

def run_algorithm(selector, matrix, num_points, labels):
    try:
        if labels is not None:
            selected_ids = selector.select(matrix, size = num_points, labels = labels)
        else:
            selected_ids = selector.select(matrix, size = num_points)

        selected_ids = [int(i) for i in selected_ids]
        st.session_state['selected_ids'] = selected_ids
        return selected_ids
    except ValueError as ve:
        st.error(f"An error occurred while running the algorithm: {ve}")
    except Exception as e:
        st.error(f"An error occurred while running the algorithm: {e}")
    return None

def export_results(selected_ids):
    export_format = st.selectbox("Select export format", ["CSV", "JSON"], key="export_format")

    if export_format == "CSV":
        csv_data = pd.DataFrame(selected_ids, columns=["Selected Indices"])
        csv = csv_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name='selected_indices.csv',
            mime='text/csv',
        )
    else:
        json_data = json.dumps({"Selected Indices": selected_ids})
        st.download_button(
            label="Download as JSON",
            data=json_data,
            file_name='selected_indices.json',
            mime='application/json',
        )
