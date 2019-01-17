import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()
from sklearn.preprocessing import StandardScaler
import pickle
import sys
sys.path.append('../../rumm')
import lang
import nets
import bayesian
import gan
import swifter
# constants
BATCH_SZ = 2048

# load_dataset
df = pd.read_csv('first_10k.dat', sep='\t', quotechar='\'', header=None, names=['0', '1'])
df['0'] = df['0'].swifter.apply(lambda x: x if len(x) <= 62 else np.nan)
df['1'] = df['1'].swifter.apply(lambda x: x if len(x) <= 62 else np.nan)
df = df.dropna()

xs = df.values[:, 0]
ys = df.values[:, 1]
xs = np.apply_along_axis(lambda x: 'G' + x + 'E', 0, xs)
ys = np.apply_along_axis(lambda x: 'G' + y + 'E', 0, xs)
xs = tf.convert_to_tensor(xs)
ys = tf.convert_to_tensor(ys)

f_handle = open('lang_obj.p', 'rb')
lang_obj = pickle.load(f_handle)
f_handle.close()
vocab_size = len(lang_obj.idx2ch) + 1

# define models
gan_box = gan.ConditionalGAN(batch_sz = 2048)
enc_f = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=False,
    enc_units = 256)
enc_b = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=True,
    enc_units = 256)
attention = nets.BidirectionalWideAttention(128)
fcuk = nets.FullyConnectedUnits([128, 'leaky_relu', 1024, 'leaky_relu', 512, 'leaky_relu', 32])
d_mean = nets.FullyConnectedUnits([16])

# initialize
xs_0 = tf.zeros([BATCH_SZ, 64])
eo_f, h_f = enc_f(xs_0)
eo_b, h_b = enc_b(xs_0)
attention_weights = attention(eo_f, eo_b, h_f, h_b)
attention_weights = fcuk(attention_weights)
mean = d_mean(attention_weights)

# load weights
enc_f.load_weights('./enc_f.h5')
enc_b.load_weights('./enc_b.h5')
attention.load_weights('./attention_weights.h5')
fcuk.load_weights('./fcuk.h5')
d_mean.load_weights('./d_mean.h5')

# load x
eo_f, h_f = enc_f(xs)
eo_b, h_b = enc_b(xs)
attention_weights = attention(eo_f, eo_b, h_f, h_b)
attention_weights = fcuk(attention_weights)
x_mean = d_mean(attention_weights)

# load y
eo_f, h_f = enc_f(ys)
eo_b, h_b = enc_b(ys)
attention_weights = attention(eo_f, eo_b, h_f, h_b)
attention_weights = fcuk(attention_weights)
y_mean = d_mean(attention_weights)

# train
gan_box.load_dataset(x_mean, y_mean)
gan_box.train()
