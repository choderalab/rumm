"""
fp.py

Handles the augmentation of the feature matrix by generating fingerprints.
"""

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
import pandas as pd

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
    return fp1

def augment_df(df, smiles_idx = 0):
    new_df = pd.DataFrame(columns = range(178))
    for idx in range(df.shape[0]):
        print(idx)
        old_line = df.loc[idx].values.tolist()
        smiles = old_line[smiles_idx]
        new_line = old_line + smiles_to_fp(smiles).tolist()
        new_df.loc[idx] = new_line
    return new_df

if __name__ = '__main__':
    import sys
    file_path = sys.argv[1]
