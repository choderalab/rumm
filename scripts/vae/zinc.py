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

# constants
BATCH_SZ = 1024

'''
zinc_df = pd.read_csv('res.csv', sep='\t')

# shuffle it, and conduct training-test split
zinc_df = zinc_df.sample(zinc_df.shape[0])
n_samples = zinc_df.shape[0]
n_tr = int(0.8 * n_samples)
y_tr = np.array(zinc_df.values[:n_tr, 2:-2], dtype=np.float32)
x_tr = zinc_df.values[:n_tr, -2]
y_te = np.array(zinc_df.values[n_tr:, 2:-2], dtype=np.float32)
x_te = zinc_df.values[n_tr:, -2]
x_tr = np.apply_along_axis(lambda x: 'G' + x + 'E', 0, x_tr)
x_te = np.apply_along_axis(lambda x: 'G' + x + 'E', 0, x_te)
fp_tr = np.array([list(line) for line in zinc_df.values[:n_tr, -1].tolist()], dtype='int')
fp_te = np.array([list(line) for line in zinc_df.values[n_tr:, -1].tolist()], dtype='int')
# calculate the std of y_tr for loss function
scaler = StandardScaler(copy=False)
y_tr = scaler.fit_transform(y_tr)
y_te = scaler.transform(y_te)
f_handle = open('scaler.p', 'wb')
pickle.dump(scaler, f_handle)
f_handle.close()

# save the dataset for later use
np.save('y_tr', y_tr)
np.save('x_tr', x_tr)
np.save('y_te', y_te)
np.save('x_te', x_te)
np.save('fp_tr', fp_tr)
np.save('fp_te', fp_te)
'''

y_tr = np.load('y_tr.npy')
x_tr = np.load('x_tr.npy')
fp_tr = np.load('fp_tr.npy')
n_samples = y_tr.shape[0]

f_handle = open('lang_obj.p', 'rb')
lang_obj = pickle.load(f_handle)
f_handle.close()
vocab_size = len(lang_obj.idx2ch) + 1

# define models
enc_f = nets.GRUEncoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=False,
    enc_units = 128)
enc_b = nets.GRUEncoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=True,
    enc_units = 128)
conv_encoder = nets.ConvEncoder(
    conv_units=[256, 512, 512],
    # pool_sizes=[8, 8, 8, 8],
    conv_kernel_sizes=[8, 12, 16],
    fcs=[128, 0.2, 'elu',
             512, 0.2, 'elu',
             512])
fcuk = nets.FullyConnectedUnits([512, 'leaky_relu', 0.25, 512])
d_mean = nets.FullyConnectedUnits([32])
d_log_var = nets.FullyConnectedUnits([32])

fcuk_props = nets.FullyConnectedUnits([9])
fcuk_fp = nets.FullyConnectedUnits([167, 'sigmoid'])
decoder = nets.OneHotDecoder(vocab_size=vocab_size, dec_units = 256)
bypass_v_f = nets.FullyConnectedUnits([1])
simple_decoder = nets.SimpleDecoder(vocab_size=vocab_size, dec_units=1024,
    batch_sz = BATCH_SZ)

# convert to tensor
x_tr = tf.convert_to_tensor(x_tr)
y_tr = tf.convert_to_tensor(y_tr)
fp_tr = tf.convert_to_tensor(fp_tr)

# make them into a dataset object
ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr, fp_tr))
ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SZ))

x_tr = None
y_tr = None
fp_tr = None
x_te = None
y_te = None
fp_te = None



optimizer = tf.train.AdamOptimizer(1e-5)
anneal_step = tf.constant(100000000.0, dtype=tf.float32)


for epoch in range(10000):
    # loop through the batches
    for (batch, (xs, ys, fps)) in enumerate(ds):
        # TODO
        # one training batch
        n_iter = tf.constant(epoch * int(n_samples) + batch * BATCH_SZ, dtype=tf.float32)
        kl_anneal = tf.cond(n_iter < anneal_step,
                            lambda: tf.math.sin(tf.div(n_iter, anneal_step) * 0.5 * tf.constant(np.pi, dtype=tf.float32)),
                            lambda: tf.constant(1.0, dtype=tf.float32))

        with tf.GradientTape() as tape: # for descent
            # training
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
            loss0 = tf.clip_by_value(tf.losses.mean_squared_error(ys, ys_hat), 0.0, 1e5)
            loss1 = tf.clip_by_value(tf.losses.log_loss(fps, fp_hat), 0.0, 1e5)
            xs_bar = decoder(z)
            loss2 = tf.clip_by_value(
                    tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(labels = xs, logits = xs_bar)), 0.0, 1e20)
            loss3 = tf.clip_by_value(kl_anneal * tf.reduce_mean(-0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=[1])), 1.0, 1e5)

            lt = loss0 + loss1 + loss2 + loss3

        # start grad norm
        variables = conv_encoder.variables +\
             d_mean.variables + decoder.variables + d_log_var.variables +\
               enc_f.variables + enc_b.variables + fcuk.variables +\
                fcuk_props.variables + fcuk_fp.variables

        gradients = tape.gradient(lt, variables)

        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

        if batch % 100 == 0:
            print(loss0.numpy(), loss1.numpy(), loss2.numpy(), loss3.numpy())

        if (batch % 1000 == 0) and ( np.isnan(lt.numpy()) == False):
            fcuk.save_weights('./fcuk.h5')
            enc_f.save_weights('./enc_f.h5')
            enc_b.save_weights('./enc_b.h5')
            fcuk_props.save_weights('./fcuk_props.h5')
            fcuk_fp.save_weights('./fcuk_fp.h5')
            d_mean.save_weights('./d_mean.h5')
            d_log_var.save_weights('./d_log_var.h5')
            decoder.save_weights('./decoder.h5')
            bypass_v_f.save_weights('./bypass_v_f.h5')
            simple_decoder.save_weights('./simple_decoder.h5')
