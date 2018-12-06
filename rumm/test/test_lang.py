"""
Test lang facility

"""

# imports
import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()

import sys
sys.path.append('..')
import lang
import nets

def test_build_vocab(file_path):
    """
    Test the construction of a vocabulary.

    """
    df = pd.read_csv(file_path)
    ys = df.values[:, -1]
    xs = df.values[:, 1]

    lang_obj = lang.Lang(xs)
    x_tensor = lang.preprocessing(xs, lang_obj)
    print(x_tensor)

test_build_vocab('data/delaney-processed.csv')
