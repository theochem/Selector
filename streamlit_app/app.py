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


# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the assets directory
assets_dir = os.path.join(current_dir, "assets")

# Set page configuration
st.set_page_config(
    page_title = "Selector",
    page_icon = os.path.join(assets_dir, "QC-Devs.png"),
)

st.image(os.path.join(assets_dir, "selector_logo.png"))

st.write("# Welcome to Selector! 👋")

st.sidebar.success("Select an algorithm to get started.")

st.info("👈 Select an algorithm from the sidebar to see some examples of what Selector can do!")

st.markdown(
    """
    [selector](https://github.com/theochem/Selector) is a free, open-source, and cross-platform
    Python library designed to help you effortlessly identify the most diverse subset of molecules
    from your dataset. 
    Please use the following citation in any publication using selector library:

    **“Selector: A Generic Python Package for Subset Selection”**, Fanwang Meng, Alireza Tehrani, 
    Valerii Chuiko, Abigail Broscius, Abdul, Hassan, Maximilian van Zyl, Marco Martínez González, 
    Yang, Ramón Alain Miranda-Quintana, Paul W. Ayers, and Farnaz Heidar-Zadeh”

    The selector source code is hosted on [GitHub](https://github.com/theochem/Selector) 
    and is released under the [GNU General Public License v3.0](https://github.com/theochem/Selector/blob/main/LICENSE). 
    We welcome any contributions to the selector library in accordance with our Code of Conduct; 
    please see our [Contributing Guidelines](https://qcdevs.org/guidelines/qcdevs_code_of_conduct/).
    Please report any issues you encounter while using 
    selector library on [GitHub Issues](https://github.com/theochem/Selector/issues). 
    For further information and inquiries please contact us at qcdevs@gmail.com.

    ### Why QC-Selector?
    In the world of chemistry, selecting the right subset of molecules is critical for a wide
    range of applications, including drug discovery, materials science, and molecular optimization. 
    QC-Selector offers a cutting-edge solution to streamline this process, empowering researchers, 
    scientists, and developers to make smarter decisions faster.

    ### Key Features
    1. Import Your Dataset: Simply import your molecule dataset in various file formats, including SDF, SMILES, and InChi, to get started.

    2. Define Selection Criteria: Specify the desired level of diversity and other relevant parameters to tailor the subset selection to your unique requirements.

    3. Run the Analysis: Let QC-Selector’s powerful algorithms process your dataset and efficiently select the most diverse molecules.

    4. Export: Explore the diverse subset and export the results for further analysis and integration into your projects.
"""
)

st.sidebar.title("About QC-Devs")

st.sidebar.info("QC-Devs develops various free, open-source, and cross-platform libraries for scientific computing, especially theoretical and computational chemistry. Our goal is to make programming accessible to chemists and promote precepts of sustainable software development. For further information and inquiries please contact us at qcdevs@gmail.com.")

# Add icons to the sidebar
st.sidebar.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <div style="text-align: center;">
        <a href="https://qcdevs.org/" target="_blank"><i class="fa fa-home" style="font-size:24px"></i> WEBSITE</a><br>
        <a href="mailto:qcdevs@gmail.com"><i class="fa fa-envelope" style="font-size:24px"></i> EMAIL</a><br>
        <a href="https://github.com/theochem" target="_blank"><i class="fa fa-github" style="font-size:24px"></i> GITHUB</a><br>
        © 2024 QC-Devs. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
