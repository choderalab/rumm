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

# constants
BATCH_SZ = 4096

'''
# load the dataset
zinc_df = pd.read_csv('../../../6_prop.xls', sep='\t')

# shuffle it, and conduct training-test split
zinc_df = zinc_df.sample(zinc_df.shape[0])
n_samples = zinc_df.shape[0]
n_tr = int(0.8 * n_samples)
y_tr = np.array(zinc_df.values[:n_tr, 1:-1], dtype=np.float32)
x_tr = zinc_df.values[:n_tr, -1]
y_te = np.array(zinc_df.values[n_tr:, 1:-1], dtype=np.float32)
x_te = zinc_df.values[n_tr:, -1]
x_tr = np.apply_along_axis(lambda x: 'G' + x + 'E', 0, x_tr)
x_te = np.apply_along_axis(lambda x: 'G' + x + 'E', 0, x_te)

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
'''

x_tr = np.load('x_tr.npy')
x_te = np.load('x_te.npy')
y_tr = np.load('y_tr.npy')
y_te = np.load('y_te.npy')

# create the language object and map it to strings
lang_obj = lang.Lang(list(x_tr) + list(x_te))
vocab_size = len(lang_obj.idx2ch) + 1
x_tr = lang.preprocessing(x_tr, lang_obj)

# define models
enc_f = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=False)
enc_b = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=True)
attention = nets.BidirectionalWideAttention(128)
fcuk = nets.FullyConnectedUnits([128, 'tanh', 0.30, 1024, 'tanh', 0.30, 1024,
'tanh', 0.25, 256, 'tanh', 0.10])
fcuk_props = nets.FullyConnectedUnits([9])
decoder = nets.AttentionDecoder(vocab_size=vocab_size)

# convert to tensor
x_tr = tf.convert_to_tensor(x_tr)
y_tr = tf.convert_to_tensor(y_tr)

# initialize
xs = tf.zeros((BATCH_SZ, 64))
eo_f, h_f = enc_f(xs)
eo_b, h_b = enc_b(xs)
attention_weights = attention(eo_f, eo_b, h_f, h_b)
attention_weights = fcuk(attention_weights)
ys_hat = fcuk_props(attention_weights)
dec_input = tf.expand_dims([lang_obj.ch2idx['G']] * BATCH_SZ, 1)
dec_hidden = decoder.initialize_hidden_state()
for t in range(xs.shape[1]):
    ch_hat, dec_hidden = decoder(dec_input, dec_hidden, attention_weights)
    dec_input = tf.expand_dims(xs[:, t], 1)

# load weights
fcuk.load_weights('./fcuk.h5')
enc_f.load_weights('./enc_f.h5')
enc_b.load_weights('./enc_b.h5')
attention.load_weights('./attention_weights.h5')
fcuk_props.load_weights('./fcuk_props.h5')
decoder.load_weights('./decoder.h5')


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
alpha = 1.5
for epoch in range(1000):
    # placeholders for the initial losses
    loss0_int = 0.0
    loss1_int = 0.0

    # for each epoch, initialize the weights for two tasks
    w0_task = tf.Variable(1.0, dtype=tf.float32)
    w1_task = tf.Variable(1.0, dtype=tf.float32)

    total_loss = 0 # initialize the total loss at the beginning to be 0
    # loop through the batches
    for (batch, (xs, ys)) in enumerate(ds):
        # the loss at the beginning of the batch is zero
        loss = 0

        # TODO
        # one training batch
        with tf.GradientTape() as tape: # for descent
            eo_f, h_f = enc_f(xs)
            eo_b, h_b = enc_b(xs)
            attention_weights = attention(eo_f, eo_b, h_f, h_b)
            attention_weights = fcuk(attention_weights)
            ys_hat = fcuk_props(attention_weights)
            loss0 = tf.losses.mean_squared_error(ys_hat, ys)
            if loss0_int == 0.0:
                loss0_int = loss0

            dec_input = tf.expand_dims([lang_obj.ch2idx['G']] * BATCH_SZ, 1)
            dec_hidden = decoder.initialize_hidden_state()
            loss1 = 0
            for t in range(xs.shape[1]):
                ch_hat, dec_hidden = decoder(dec_input, dec_hidden, attention_weights)
                loss1 += seq_loss(xs[:, t], ch_hat)
                dec_input = tf.expand_dims(xs[:, t], 1)
            if loss1_int == 0.0:
                loss1_int = loss1


        # start grad norm
        loss0_var = enc_f.variables + enc_b.variables + attention.variables + fcuk_props.variables
        loss1_var = decoder.variables
        lt = w0_task * loss0 + w1_task * loss1
        gw0 = w0_task * np.norm(tape.gradient(loss0, loss0_var))
        gw1 = w1_task * np.norm(tape.gradient(loss1, loss1_var))
        gw_bar = 0.5 * (gw0 + gw1)
        l0_tilde = np.true_divide(loss0, loss0_int)
        l1_tilde = np.true_divide(loss1, loss1_int)
        l_tilde_bar = 0.5 * (l0_tilde + l1_tilde)
        r0 = np.true_divide(l0_tilde, l_tilde_bar)
        r1 = np.true_divide(t1_tilde, l_tilde_bar)

        l_grad = np.abs(gw0 - gw_bar * np.power(r0, alpha)) +\
                 np.abs(gw1 - gw_bar * np.power(r1, alpha))

        delta_l_grad_0 = tape.gradient(l_grad, w0_task)
        delta_l_grad_1 = tape.gradient(l_grad, w1_task)

        optimizer.apply_gradients(zip(delta_l_grad_0, [w0_task]))
        optimizer.apply_gradients(zip(delta_l_grad_1, [w1_task]))

        variables = enc_f.variables + enc_b.variables + attention.variables + fcuk_props.variables + decoder.variables + fcuk.variables

        gradients = tape.gradient(l_grad, variables)
        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
        if batch % 10 == 0:
            print("epoch %s batch %s loss %s" % (epoch, batch, np.asscalar(l_grad.numpy())))

        fcuk.save_weights('./fcuk.h5')
        enc_f.save_weights('./enc_f.h5')
        enc_b.save_weights('./enc_b.h5')
        attention.save_weights('./attention_weights.h5')
        fcuk_props.save_weights('./fcuk_props.h5')
        decoder.save_weights('./decoder.h5')
