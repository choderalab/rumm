import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()
from sklearn.preprocessing import StandardScaler
import pickle
import sys
sys.path.append('..')
import lang
import nets
import bayesian

# load dataset
x_te = np.load('x_te.npy')
y_te = np.load('y_te.npy')
x_tr = np.load('x_tr.npy')
y_tr = np.load('y_tr.npy')

# create the language object and map it to strings
lang_obj = lang.Lang(list(x_tr) + list(x_te))
vocab_size = len(lang_obj.idx2ch) + 1
x_te = lang.preprocessing(x_te, lang_obj)
x_tr = lang.preprocessing(x_tr, lang_obj)

# convert to tensor
x_tr = tf.convert_to_tensor(x_tr)
y_tr = tf.convert_to_tensor(y_tr)

# define models
enc_f = nets.Encoder(vocab_size=vocab_size, batch_sz = 1, reverse=False)
enc_b = nets.Encoder(vocab_size=vocab_size, batch_sz = 1, reverse=True)
attention = nets.BidirectionalWideAttention(128)
fcuk = nets.FullyConnectedUnits([512, 'tanh', 0.30, 512, 'tanh', 0.30, 512, 'tanh', 0.25])
fcuk_props = nets.FullyConnectedUnits([9])
decoder = nets.AttentionDecoder(vocab_size=vocab_size)

# initialize
xs = tf.zeros((1, vocab_size))
eo_f, h_f = enc_f(xs)
eo_b, h_b = enc_b(xs)
attention_weights = attention(eo_f, eo_b, h_f, h_b)
attention_weights = fcuk(attention_weights)
ys_hat = fcuk_props(attention_weights)

# load weights
fcuk.load_weights('./fcuk.h5')
enc_f.load_weights('./enc_f.h5')
enc_b.load_weights('./enc_b.h5')
attention.load_weights('./attention_weights.h5')
fcuk_props.load_weights('./fcuk_props.h5')
decoder.load_weights('./decoder.h5')
