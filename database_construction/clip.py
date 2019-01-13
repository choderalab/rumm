"""
fp.py

Handles the augmentation of the feature matrix by generating fingerprints.
"""

import pandas as pd
import swifter
import numpy as np


if __name__ == '__main__':
    import sys
    file_path = sys.argv[1]
    df = pd.read_csv(file_path, sep='\t')
    df['SMILES'] = df['SMILES'].swifter.apply(lambda x: x if len(x) <= 64 else np.nan)
    df = df.dropna()
    df.to_csv('res1.csv', sep='\t')
