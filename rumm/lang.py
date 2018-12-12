"""
lang.py

Handles the transformation of smiles strings.
"""
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

class Lang:
    """
    Translates the characters of smiles into a index representation.


    """
    def __init__(self, lang):
        self.lang = lang
        self.ch2idx = {}
        self.idx2ch = {}
        self.vocab = set()
        self.create_idx()
        self.lang = None

    def create_idx(self):
        # exhaust the vocabulary in the smiles collection
        for smiles in self.lang:
            self.vocab.update(list(smiles))

        self.vocab = sorted(self.vocab)

        for idx, ch in enumerate(self.vocab):
            self.ch2idx[ch] = idx + 1

        # put 'G', 'E', and '0' in the vocabulary
        # signifying 'GO', 'END', and padding.
        self.ch2idx['G'] = idx + 2
        self.ch2idx['E'] = idx + 3
        self.ch2idx['0'] = 0

        # build the mapping in both way
        for ch, idx in self.ch2idx.items():
            self.idx2ch[idx] = ch


def preprocessing(smiles_array, lang, max_len = 64):
    """
    Preprocess a smiles into a continuous matrix representation.

    """
    x_tensor = [[lang.ch2idx[ch] for ch in list(smiles)] for smiles in smiles_array]
    x_tensor = tf.keras.preprocessing.sequence.pad_sequences(x_tensor,
                                                                 maxlen=max_len,
                                                                 padding='post')
    return x_tensor
