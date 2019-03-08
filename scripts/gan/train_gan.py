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
df = pd.read_csv('all_chembl_mol_pairs.dat', sep='\t', quotechar='\'', header=None, names=['0', '1'])
df['0'] = df['0'].swifter.apply(lambda x: x if len(x) <= 62 and 'i' not in
    x and 'e' not in x and 'A' not in x else np.nan)
df['1'] = df['1'].swifter.apply(lambda x: x if len(x) <= 62 and 'i' not in
    x and 'e' not in x and 'A' not in x else np.nan)
df = df.dropna()

xs = df.values[:, 0]
ys = df.values[:, 1]
# xs = np.apply_along_axis(lambda x: 'G' + x + 'E', 0, xs)
# ys = np.apply_along_axis(lambda x: 'G' + x + 'E', 0, ys)
f_handle = open('lang_obj.p', 'rb')
lang_obj = pickle.load(f_handle)
f_handle.close()
vocab_size = len(lang_obj.idx2ch) + 1
xs = lang.preprocessing(xs, lang_obj)
ys = lang.preprocessing(ys, lang_obj)
xs = tf.convert_to_tensor(xs)
ys = tf.convert_to_tensor(ys)

# define models
gan_box = gan.ConditionalGAN(batch_sz = 2048)
# define models
enc_f = nets.GRUEncoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=False,
    enc_units = 512)
enc_b = nets.GRUEncoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=True,
    enc_units = 512)
conv_encoder = nets.ConvEncoder(
    conv_units=[256, 512, 512],
    # pool_sizes=[8, 8, 8, 8],
    conv_kernel_sizes=[8, 12, 16],
    fcs=[128, 0.2, 'elu',
             512, 0.2, 'elu',
             512])
fcuk = nets.FullyConnectedUnits([512, 'leaky_relu', 0.25, 512])
d_mean = nets.FullyConnectedUnits([256])
d_log_var = nets.FullyConnectedUnits([256])

fcuk_props = nets.FullyConnectedUnits([9])
fcuk_fp = nets.FullyConnectedUnits([167, 'sigmoid'])
decoder = nets.OneHotDecoder(vocab_size=vocab_size, dec_units = 512)


# initialize
xs = tf.zeros([BATCH_SZ, 64], dtype=tf.int64)
eo_f, h_f = enc_f(xs)
eo_b, h_b = enc_b(xs)
x_attention = tf.concat([h_f, h_b], axis=-1)
x_attention = fcuk(x_attention)
x_conv = conv_encoder(tf.one_hot(xs, 33))
x = tf.concat([x_attention, x_conv], axis=-1)
mean = d_mean(x)
log_var = d_log_var(x)
z_noise = tf.clip_by_norm(tf.random_normal(mean.shape), 1e5) * tf.exp(log_var * .5)
z = z_noise + mean
ys_hat = fcuk_props(mean)
fp_hat = fcuk_fp(mean)
xs_bar = decoder(z)

# load weights
enc_f.load_weights('weights/enc_f.h5')
enc_b.load_weights('weights/enc_b.h5')
conv_encoder.load_weights('weights/conv_encoder.h5')
fcuk.load_weights('weights/fcuk.h5')
d_mean.load_weights('weights/d_mean.h5')
d_log_var.load_weights('weights/d_log_var.h5')
fcuk_props.load_weights('weights/fcuk_props.h5')
fcuk_fp.load_weights('weights/fcuk_fp.h5')
# bypass_v_f.load_weights('weights/bypass_v_f.h5')
decoder.load_weights('weights/decoder.h5')

ds = tf.data.Dataset.from_tensor_slices((xs, ys))
ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SZ))
for (batch, (xs, ys)) in enumerate(ds):
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

    if batch == 0:
        x_out = x_mean
        y_out = y_mean

    else:
        x_out = tf.concat([x_out, x_mean], axis=0)
        y_out = tf.concat([y_out, y_mean], axis=0)

np.save('x_mean', x_out.numpy())
np.save('y_mean', y_out.numpy())
# train
gan_box.load_dataset(x_mean, y_mean)
gan_box.train()
