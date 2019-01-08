"""
fp.py

Handles the augmentation of the feature matrix by generating fingerprints.
"""

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
import pandas as pd
import swifter

def smiles_to_fp(smiles):
    """
    Convert smiles to Daylight FP and MACCSkeys.

    Parameters
    ----------
    smiles : str, smiles representation of a molecule

    Returns
    -------
    fp : np.ndarray zero and one representation of the fingerprints
    """
    mol = Chem.MolFromSmiles(smiles)
    fp1 = MACCSkeys.GenMACCSKeys(mol)
    fp1 = np.fromstring(fp1.ToBitString(), 'int8') - 48
    fp1 = fp1.tolist()
    return fp1

if __name__ = '__main__':
    import sys
    file_path = sys.argv[1]
    df = pd.read_csv(file_path, sep='\t')
    df[range(166)] = df.swifter.apply(smiles_to_fp)
