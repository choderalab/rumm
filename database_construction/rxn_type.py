"""
rxn_type.py

screen through database built from ChEMBL to look for certain reaction types.
"""
# imports
import numpy
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
