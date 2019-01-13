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

# utility functions

class RxnIdentifier:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_rxn(self):
                                                                  

    def is_rxn(self, smiles0, smiles1):
        """
        Judge weather smiles1 is a possible transformation from smiles0 through
        given reaction type expressed by rxn_smarts.

        Parameters
        ----------
        smiles0 : str, a smiles expression of a molecule
        smiles1 : str, a smiles expression of a molecule
        rxn_smirks : smirks expression of a reaction
        """

        # transform smiles to molecule objects
        mol0 = Chem.MolFromSmiles(smiles0)
        mol1 = Chem.MolFromSmiles(smiles1)

        # load the reaction
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
