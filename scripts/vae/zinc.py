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
BATCH_SZ = 512

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

f_handle = open('lang_obj.p', 'rb')
lang_obj = pickle.load(f_handle)
f_handle.close()
vocab_size = len(lang_obj.idx2ch) + 1

# define models
enc_f = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=False,
    enc_units = 256)
enc_b = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=True,
    enc_units = 256)
attention = nets.BidirectionalWideAttention(128)
fcuk = nets.FullyConnectedUnits([512, 'leaky_relu', 0.25, 512])
d_mean = nets.FullyConnectedUnits([16])
d_log_var = nets.FullyConnectedUnits([16])
fcuk_props = nets.FullyConnectedUnits([9])
fcuk_fp = nets.FullyConnectedUnits([167, 'sigmoid'])
decoder = nets.OneHotDecoder(vocab_size=vocab_size, dec_units = 128)
bypass_v_f = nets.FullyConnectedUnits([1])

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

optimizer = tf.train.AdamOptimizer(1e-3)
alpha = 0.5

for epoch in range(1000):
    # loop through the batches
    for (batch, (xs, ys, fps)) in enumerate(ds):
        # TODO
        # one training batch
        apply_norm = (batch % 10 == 0)
        n_iter = tf.constant(epoch * int(xs.shape[0]) + batch * BATCH_SZ, dtype=tf.float32)
        anneal_step = tf.constant(20000000.0, dtype=tf.float32)

        with tf.GradientTape(persistent=apply_norm) as tape: # for descent
            # training
            kl_anneal = tf.cond(n_iter < anneal_step,
                                lambda: tf.math.sin(tf.div(n_iter, anneal_step) * 0.5 * tf.constant(np.pi, dtype=tf.float32)),
                                lambda: tf.constant(1.0, dtype=tf.float32))

            eo_f, h_f = enc_f(xs)
            eo_b, h_b = enc_b(xs)

            bypass_xs_f = tf.layers.flatten(bypass_v_f(eo_f))

            attention_weights = attention(eo_f, eo_b, h_f, h_b)
            x = fcuk(attention_weights)
            x = tf.concat([x, bypass_xs_f], axis=-1)

            mean = d_mean(x)
            log_var = d_log_var(x)
            z = tf.clip_by_norm(tf.random_normal(mean.shape), 1e5) * tf.exp(log_var * .5) * kl_anneal + mean

            ys_hat = fcuk_props(mean)
            fp_hat = fcuk_fp(mean)

            loss0 = tf.clip_by_value(tf.losses.mean_squared_error(ys, ys_hat),
                                     0.0, 1e5)
            loss1 = tf.clip_by_value(tf.losses.mean_squared_error(fps, fp_hat),
                                     0.0, 1e5)

            xs_bar = decoder(z)

            loss2 = tf.clip_by_value(
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = xs, logits = xs_bar)), 0.0, 1e5)
            loss3 = tf.clip_by_value(kl_anneal * tf.reduce_mean(-0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var))),
                                     0.0, 1e5)

            lt = w0_task * loss0 + w1_task * loss1 + w2_task * loss2 + w3_task * loss3

        # start grad norm
        variables = enc_f.variables + enc_b.variables + fcuk.variables +\
                    attention.variables + fcuk_props.variables + decoder.variables + fcuk_fp.variables +\
                    d_mean.variables + d_log_var.variables + bypass_v_f.variables

        gradients = tape.gradient(lt, variables)

        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

        if apply_norm:
            if batch % 100 == 0: # update the initial loss every 100 batches
                loss0_int = loss0
                loss1_int = loss1
                loss2_int = loss2
                loss3_int = loss3

                print("epoch %s batch %s loss %s" % (epoch, batch, np.asscalar(lt.numpy())))
                print(loss0.numpy(), loss1.numpy(), loss2.numpy(), loss3.numpy())

            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape1:
                tape1.watch(w0_task)
                tape1.watch(w1_task)
                tape1.watch(w2_task)
                tape1.watch(w3_task)

                gw_grad_0 = w0_task * tf.clip_by_norm(tape.gradient(loss0, d_mean.C0.weights[0]), 1e2)
                gw_grad_1 = w1_task * tf.clip_by_norm(tape.gradient(loss1, d_mean.C0.weights[0]), 1e2)
                gw_grad_2 = w2_task * tf.clip_by_norm(tape.gradient(loss2, d_mean.C0.weights[0]), 1e2)
                gw_grad_3 = w3_task * tf.clip_by_norm(tape.gradient(loss3, d_mean.C0.weights[0]), 1e2)

                gw0 = tf.clip_by_norm(tf.norm(gw_grad_0), 1e2)
                gw1 = tf.clip_by_norm(tf.norm(gw_grad_1), 1e2)
                gw2 = tf.clip_by_norm(tf.norm(gw_grad_2), 1e2)
                gw3 = tf.clip_by_norm(tf.norm(gw_grad_3), 1e2)

                gw_bar = tf.div_no_nan(gw0 + gw1 + gw2 + gw3, 4.0)

                l0_tilde = tf.clip_by_norm(tf.div_no_nan(loss0, loss0_int),
                    1e5)
                l1_tilde = tf.clip_by_norm(tf.div_no_nan(loss1, loss1_int),
                    1e5)
                l2_tilde = tf.clip_by_norm(tf.div_no_nan(loss2, loss2_int),
                    1e5)
                l3_tilde = tf.clip_by_norm(tf.div_no_nan(loss3, loss3_int),
                    1e5)

                l_tilde_bar = tf.div_no_nan(l0_tilde + l1_tilde + l2_tilde + l3_tilde, 4.0)

                r0 = tf.div_no_nan(l0_tilde, l_tilde_bar)
                r1 = tf.div_no_nan(l1_tilde, l_tilde_bar)
                r2 = tf.div_no_nan(l2_tilde, l_tilde_bar)
                r3 = tf.div_no_nan(l3_tilde, l_tilde_bar)

                l_grad = tf.math.abs(gw0 - tf.stop_gradient(gw_bar * tf.math.pow(r0, alpha))) +\
                         tf.math.abs(gw1 - tf.stop_gradient(gw_bar * tf.math.pow(r1, alpha))) +\
                         tf.math.abs(gw2 - tf.stop_gradient(gw_bar * tf.math.pow(r2, alpha))) +\
                         tf.math.abs(gw3 - tf.stop_gradient(gw_bar * tf.math.pow(r3, alpha)))


                l_grad = tf.clip_by_norm(l_grad, 1e2)

            delta_l_grad_0 = tf.clip_by_norm(tape1.gradient(l_grad, w0_task),
                1e2)
            delta_l_grad_1 = tf.clip_by_norm(tape1.gradient(l_grad, w1_task),
                1e2)
            delta_l_grad_2 = tf.clip_by_norm(tape1.gradient(l_grad, w2_task),
                1e2)
            delta_l_grad_3 = tf.clip_by_norm(tape1.gradient(l_grad, w3_task),
                1e2)

            del tape
            del tape1

            @tf.contrib.eager.defun
            def update():
                optimizer.apply_gradients([(delta_l_grad_0, w0_task)])
                optimizer.apply_gradients([(delta_l_grad_1, w1_task)])
                optimizer.apply_gradients([(delta_l_grad_2, w2_task)])
                optimizer.apply_gradients([(delta_l_grad_3, w3_task)])

                w0_task.assign(tf.clip_by_value(w0_task, 0.5, 2.0))
                w1_task.assign(tf.clip_by_value(w1_task, 0.5, 2.0))
                w2_task.assign(tf.clip_by_value(w2_task, 0.5, 2.0))
                w3_task.assign(tf.clip_by_value(w3_task, 0.5, 2.0))

                w_total = w0_task + w1_task + w2_task + w3_task
                w0_task.assign(w0_task * tf.div_no_nan(4.0, w_total))
                w1_task.assign(w1_task * tf.div_no_nan(4.0, w_total))
                w2_task.assign(w2_task * tf.div_no_nan(4.0, w_total))
                w3_task.assign(w3_task * tf.div_no_nan(4.0, w_total))

            tf.cond(tf.debugging.is_nan(delta_l_grad_0 + delta_l_grad_1 + delta_l_grad_2 + delta_l_grad_3),
                   lambda: None,
                   lambda: update())

            if (batch % 1000 == 0) and ( np.isnan(lt.numpy()) == False):
                fcuk.save_weights('./fcuk.h5')
                enc_f.save_weights('./enc_f.h5')
                enc_b.save_weights('./enc_b.h5')
                attention.save_weights('./attention_weights.h5')
                fcuk_props.save_weights('./fcuk_props.h5')
                fcuk_fp.save_weights('./fcuk_fp.h5')
                d_mean.save_weights('./d_mean.h5')
                d_log_var.save_weights('./d_log_var.h5')
                decoder.save_weights('./decoder.h5')
                bypass_v_f.save_weights('./bypass_v_f.h5')
