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
BATCH_SZ = 4096

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
# fcuk_props = nets.FullyConnectedUnits([9])
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
# fcuk_props.load_weights('./fcuk_props.h5')
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
for epoch in range(1000):
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

            dec_input = tf.expand_dims([lang_obj.ch2idx['G']] * BATCH_SZ, 1)
            dec_hidden = decoder.initialize_hidden_state()
            loss1 = 0
            for t in range(xs.shape[1]):
                ch_hat, dec_hidden = decoder(dec_input, dec_hidden, attention_weights)
                loss1 += seq_loss(xs[:, t], ch_hat)
                dec_input = tf.expand_dims(xs[:, t], 1)
            loss += loss1

        total_loss += loss
        variables = enc_f.variables + enc_b.variables + attention.variables + decoder.variables + fcuk.variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
        if batch % 10 == 0:
            print("epoch %s batch %s loss %s" % (epoch, batch, np.asscalar(loss.numpy())))

        fcuk.save_weights('./fcuk.h5')
        enc_f.save_weights('./enc_f.h5')
        enc_b.save_weights('./enc_b.h5')
        attention.save_weights('./attention_weights.h5')
        decoder.save_weights('./decoder.h5')
