import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()
from sklearn.preprocessing import StandardScaler
import pickle
import sys
sys.path.append('../rumm')
import lang
import nets
import bayesian

# constants
BATCH_SZ = 4096
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

# create the language object and map it to strings
lang_obj = lang.Lang(list(x_tr) + list(x_te))
vocab_size = len(lang_obj.idx2ch) + 1
x_tr = lang.preprocessing(x_tr, lang_obj)

# define models
enc_f = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=False, enc_units = 256, embedding_dim = 8)
enc_b = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=True, enc_units = 256, embedding_dim = 8)
attention = nets.BidirectionalWideAttention(128)
fcuk = nets.FullyConnectedUnits([128, 'leaky_relu', 0.10, 512, 'leaky_relu', 0.25, 512, 'leaky_relu'])
d_mean = nets.FullyConnectedUnits([16])
d_log_var = nets.FullyConnectedUnits([16])
fcuk_props = nets.FullyConnectedUnits([9])
fcuk_fp = nets.FullyConnectedUnits([167, 'sigmoid'])
decoder = nets.DeepAttentionDecoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, embedding_dim = 8)

# convert to tensor
x_tr = tf.convert_to_tensor(x_tr)
y_tr = tf.convert_to_tensor(y_tr)

# make them into a dataset object
ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(y_tr.shape[0])
ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SZ))

# get your favorite optimizer
optimizer=tf.train.AdamOptimizer()

# define loss function for sequence
def seq_loss(y, y_hat):
    mask = 1 - np.equal(y, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat) * mask
    return tf.reduce_mean(loss_)

# train it!
# loop through the epochs

w0_task = tf.Variable(1.0, dtype=tf.float32)
w1_task = tf.Variable(1.0, dtype=tf.float32)
w2_task = tf.Variable(1.0, dtype=tf.float32)
w3_task = tf.Variable(1.0, dtype=tf.float32)


alpha = 0.5
for epoch in range(1000):

    total_loss = 0 # initialize the total loss at the beginning to be 0
    # loop through the batches
    for (batch, (xs, ys, fps)) in enumerate(ds):
        # TODO
        # one training batch
        with tf.GradientTape(persistent=True) as tape: # for descent
            # training
            eo_f, h_f = enc_f(xs)
            eo_b, h_b = enc_b(xs)
            attention_weights = attention(eo_f, eo_b, h_f, h_b)
            attention_weights = fcuk(attention_weights)

            mean = d_mean(attention_weights)
            log_var = d_log_var(attention_weights)

            z = tf.random_normal(mean.shape) * tf.exp(log_var * .5) + mean

            ys_hat = fcuk_props(mean)
            fp_hat = fcuk_fp(mean)

            loss0 = tf.losses.mean_squared_error(ys, ys_hat)
            loss1 = tf.losses.mean_squared_error(fps, fp_hat)
            dec_input = tf.expand_dims([lang_obj.ch2idx['G']] * BATCH_SZ, 1)
            hidden = decoder.initialize_hidden_state()
            loss2 = 1e-8
            for t in range(xs.shape[1]):
                ch_hat, hidden = decoder(dec_input, z, hidden)
                loss2 += seq_loss(xs[:, t], ch_hat)
                dec_input = tf.expand_dims(xs[:, t], 1)

            loss3 = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1))


            if batch == 0:
                loss0_int = loss0
                loss1_int = loss1
                loss2_int = loss2
                loss3_int = loss3

            lt = w0_task * loss0 + w1_task * loss1 + w2_task * loss2 + w3_task * loss3

        # start grad norm
        variables = enc_f.variables + enc_b.variables + fcuk.variables +\
                    attention.variables + fcuk_props.variables + decoder.variables + fcuk_fp.variables +\
                    d_mean.variables + d_log_var.variables

        gradients = tape.gradient(lt, variables)

        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

        if batch % 10 == 0:
            print("epoch %s batch %s loss %s" % (epoch, batch, np.asscalar(lt.numpy())))
            print(loss0.numpy(), loss1.numpy(), loss2.numpy(), loss3.numpy())

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape1:
            tape1.watch(w0_task)
            tape1.watch(w1_task)
            tape1.watch(w2_task)
            tape1.watch(w3_task)

            gw_grad_0 = w0_task * tape.gradient(loss0, d_mean.C0.weights[0])
            gw_grad_1 = w1_task * tape.gradient(loss1, d_mean.C0.weights[0])
            gw_grad_2 = w2_task * tape.gradient(loss2, d_mean.C0.weights[0])
            gw_grad_3 = w3_task * tape.gradient(loss3, d_mean.C0.weights[0])

            gw0 = tf.clip_by_value(tf.norm(gw_grad_0), -1e18, 1e18)
            gw1 = tf.clip_by_value(tf.norm(gw_grad_1), -1e18, 1e18)
            gw2 = tf.clip_by_value(tf.norm(gw_grad_2), -1e18, 1e18)
            gw3 = tf.clip_by_value(tf.norm(gw_grad_3), -1e18, 1e18)

            gw_bar = tf.div_no_nan(gw0 + gw1 + gw2 + gw3, 4.0)
            l0_tilde = tf.div_no_nan(loss0, loss0_int)
            l1_tilde = tf.div_no_nan(loss1, loss1_int)
            l2_tilde = tf.div_no_nan(loss2, loss2_int)
            l3_tilde = tf.div_no_nan(loss3, loss3_int)

            l_tilde_bar = tf.div_no_nan(l0_tilde + l1_tilde + l2_tilde + l3_tilde, 4.0)

            r0 = tf.div_no_nan(l0_tilde, l_tilde_bar)
            r1 = tf.div_no_nan(l1_tilde, l_tilde_bar)
            r2 = tf.div_no_nan(l2_tilde, l_tilde_bar)
            r3 = tf.div_no_nan(l3_tilde, l_tilde_bar)

            l_grad = tf.math.abs(gw0 - tf.stop_gradient(gw_bar * tf.math.pow(r0, alpha))) +\
                     tf.math.abs(gw1 - tf.stop_gradient(gw_bar * tf.math.pow(r1, alpha))) +\
                     tf.math.abs(gw2 - tf.stop_gradient(gw_bar * tf.math.pow(r2, alpha))) +\
                     tf.math.abs(gw3 - tf.stop_gradient(gw_bar * tf.math.pow(r3, alpha)))


        delta_l_grad_0 = tf.clip_by_norm(tape1.gradient(l_grad, w0_task), 1e18)
        delta_l_grad_1 = tf.clip_by_norm(tape1.gradient(l_grad, w1_task), 1e18)
        delta_l_grad_2 = tf.clip_by_norm(tape1.gradient(l_grad, w2_task), 1e18)
        delta_l_grad_3 = tf.clip_by_norm(tape1.gradient(l_grad, w3_task), 1e18)

        optimizer.apply_gradients([(delta_l_grad_0, w0_task)])
        optimizer.apply_gradients([(delta_l_grad_1, w1_task)])
        optimizer.apply_gradients([(delta_l_grad_2, w2_task)])
        optimizer.apply_gradients([(delta_l_grad_3, w3_task)])

        w_total = w0_task + w1_task + w2_task + w3_task

        w0_task.assign(w0_task * tf.div_no_nan(4.0, w_total))
        w1_task.assign(w1_task * tf.div_no_nan(4.0, w_total))
        w2_task.assign(w2_task * tf.div_no_nan(4.0, w_total))
        w3_task.assign(w3_task * tf.div_no_nan(4.0, w_total))

    if not tf.debugging.is_nan(lt):
        fcuk.save_weights('./fcuk.h5')
        enc_f.save_weights('./enc_f.h5')
        enc_b.save_weights('./enc_b.h5')
        attention.save_weights('./attention_weights.h5')
        fcuk_props.save_weights('./fcuk_props.h5')
        fcuk_fp.save_weights('./fcuk_fp.h5')
        d_mean.save_weights('./d_mean.h5')
        d_log_var.save_weights('./d_log_var.h5')
        decoder.save_weights('./decoder.h5')
