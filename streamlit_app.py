# Import necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors


# Function to calculate the aromatic proportion of a molecule
def AromaticProportion(mol):
    aromatic_atoms = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in range(mol.GetNumAtoms())]
    aa_count = sum(1 for atom in aromatic_atoms if atom)
    heavy_atom_count = Descriptors.HeavyAtomCount(mol)
    aromatic_proportion = aa_count / heavy_atom_count if heavy_atom_count > 0 else 0
    return aromatic_proportion


# Function to generate molecular descriptors from SMILES strings
def generate_descriptors(smiles_list, verbose=False):
    mol_data = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_data.append(mol)

    base_data = np.array([[]])
    for i, mol in enumerate(mol_data):
        mol_logp = Descriptors.MolLogP(mol)
        mol_wt = Descriptors.MolWt(mol)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_proportion = AromaticProportion(mol)

        row = np.array([mol_logp, mol_wt, num_rotatable_bonds, aromatic_proportion])

        if i == 0:
            base_data = row
        else:
            base_data = np.vstack([base_data, row])

    column_names = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"]
    descriptors = pd.DataFrame(data=base_data, columns=column_names)

    return descriptors


# Load the logo image
image = Image.open('logo.jpg')

# Set the page title and display the logo
st.image(image, use_column_width=True)
st.write("""
# Molecular Solubility Prediction Web App

This app accepts a molecule's SMILES notation, computes its descriptors using the RDKit library, and utilizes a trained model for solubility predictions.
""")

# Sidebar for user input
st.sidebar.header('User Input Features')

# Read SMILES input
SMILES_input = "NCCCC\nCCC\nCN"
SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = "C\n" + SMILES  # Adds a dummy 'C' as the first item
SMILES = SMILES.split('\n')

# Display the input SMILES
st.header('Input SMILES')
st.write(SMILES[1:])  # Skip the dummy first item

# Calculate molecular descriptors
st.header('Computed Molecular Descriptors')
descriptors = generate_descriptors(SMILES)
st.write(descriptors[1:])  # Skip the dummy first item

# Load the saved model
model = pickle.load(open('LogS_Prediction_Model.pkl', 'rb'))

# Make predictions using the loaded model
predictions = model.predict(descriptors)

# Display the predicted LogS values
st.header('Predicted LogS Values')
st.write(predictions[1:])  # Skip the dummy first item
