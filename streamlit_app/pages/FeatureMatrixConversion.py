import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import tempfile
import os
import sys


# Add the streamlit_app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(module_path)
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

from utils import *
from streamlit_app.features import DescriptorGenerator, FingerprintGenerator


# Set up the page configuration
set_page_config(
    page_title = "Chem Converter",
    page_icon = os.path.join(parent_dir, "assets", "QC-Devs.png")
)

st.title("Chemical File Converter")

# Description of the page
st.markdown("""
This page allows you to upload raw chemical file formats such as SMILES or SDF,
and convert them into chemical matrices that can be used as input for selector's various algorithms.
""")

# File uploader for chemical file
chemical_file = st.file_uploader("Upload a chemical file (e.g., SMILES, SDF, or TXT)",
                                 type = ["txt", "smi", "sdf"])

if chemical_file:
    # User selects the file format
    file_format = st.selectbox(
        "Select the format of the provided file",
        options = ["", "SMILES", "SDF"]
    )

    if file_format:
        molecules = []
        temp_sdf_path = None

        # Process the chemical file based on user selection
        if file_format == "SMILES":
            smiles_list = chemical_file.read().decode("utf-8").splitlines()
            molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        elif file_format == "SDF":
            # Create a temporary file to save the uploaded SDF content
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".sdf") as temp_sdf:
                temp_sdf.write(chemical_file.read())
                temp_sdf_path = temp_sdf.name

            # Use RDKit's SDMolSupplier to read molecules from the SDF file
            supplier = Chem.SDMolSupplier(temp_sdf_path)
            molecules = [mol for mol in supplier if mol is not None]

            # Explicitly close the supplier to release the file
            del supplier

        # Check for valid molecules
        valid_molecules = [mol for mol in molecules if mol is not None]

        if not valid_molecules:
            st.error("No valid molecules found in the uploaded file.")
        else:
            st.success(f"Successfully loaded {len(valid_molecules)} valid molecules.")

            # Display the molecules
            img = Draw.MolsToImage(valid_molecules)
            st.image(img, caption = "Molecules in the file")

            # Choose the type of matrix to generate
            matrix_type = st.selectbox("Choose matrix type", ["Descriptors", "Fingerprints"])

            if matrix_type == "Descriptors":
                # Allow the user to choose the type of descriptors to generate
                use_fragment = st.checkbox("Whether return value includes the fragment binary descriptors", value = True)
                ipc_avg = st.checkbox("Whether IPC descriptor calculates with avg", value = True)

                descriptor_generator = DescriptorGenerator(valid_molecules)
                matrix = descriptor_generator.rdkit_desc(use_fragment, ipc_avg)
            elif matrix_type == "Fingerprints":
                # Allow user to choose the type of fingerprint to generate
                fp_type = st.selectbox("Select Fingerprint Type", options=["SECFP", "ECFP", "Morgan"])
                n_bits = st.number_input("Number of bits for the fingerprint", min_value = 1, value = 2048)
                radius = st.number_input("The maximum radius of the substructure that is generated at each atom", min_value = 1, value = 3)
                min_radius = st.number_input("The minimum radius that is used to extract n-grams", min_value = 1, value = 3)
                random_seed = st.number_input("Random seed for fingerprint generation", min_value = 0, value = 12345)
                rings = st.checkbox("Whether the rings (SSSR) are extracted from the molecule and added to the shingling", value = True)
                isomeric = st.checkbox("Whether the SMILES added to the shingling are isomeric", value = True)
                kekulize = st.checkbox("Whether the SMILES added to the shingling are kekulized", value = False)

                fp_generator = FingerprintGenerator(valid_molecules)
                matrix = fp_generator.compute_fingerprint(fp_type = fp_type)

            st.write("Generated Chemical Matrix:")
            st.dataframe(matrix)

            # Option to download the matrix as CSV
            csv_data = matrix.to_csv().encode('utf-8')
            st.download_button("Download Chemical Matrix as CSV", data = csv_data,
                               file_name = "chemical_matrix.csv", mime = "text/csv")

        # Clean up the temporary file after RDKit is done with it
        if temp_sdf_path and os.path.exists(temp_sdf_path):
            os.remove(temp_sdf_path)
