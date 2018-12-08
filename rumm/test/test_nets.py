"""
Test the neural nets models.

"""

# imports
import tensorflow as tf
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
import lang
import nets

def test_single_task_training(file_path):
    # load dataset
    df = pd.read_csv(file_path)
    y_tr = df.values[:, 1]
    x_tr = df.values[:, -1]
    lang_obj = lang.Lang(x_tr)
    vocab_size = len(lang_obj.idx2ch)
    x_tensor = lang.preprocessing(x_tr, lang_obj)

    # define models
    enc_f = nets.Encoder(vocab_size=vocab_size, reverse=False)
    enc_b = nets.Encoder(vocab_size=vocab_size, reverse=True)
    attention = nets.BidirectionalAttention(128)
    fcuk = nets.FullyConnectedUnits([128, 'tanh', 0.25,
                                     32, 'tanh', 0.10,
                                     1])

    # define the flow function
    def flow(xs, models):
        enc_f, enc_b, attention, fcuk = models
        eo_f, h_f = enc_f(xs)
        eo_b, h_b = enc_b(xs)
        attention_weights = attention(eo_f, eo_b, h_f, h_b)
        ys = fcuk(attention_weights)
        return ys

    box = nets.Box(flow=flow,
                   models=[enc_f, enc_b, attention, fcuk],
                   n_epochs=1,
                   batch_sz=32)

    box.train(x_tensor, y_tr)
    box.save_weights('box')

test_single_task_training('data/delaney-processed.csv')
