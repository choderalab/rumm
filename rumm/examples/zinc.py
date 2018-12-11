import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('..')
import lang
import nets
import bayesian

# constants
BATCH_SZ = 64

# load the dataset
zinc_df = pd.read_csv('6_prop.xls', sep='\t')

# shuffle it, and conduct training-test split
zinc_df = zinc_df.sample(zinc_df.shape[0])
n_samples = zinc_df.shape[0]
n_tr = int(0.8 * n_samples)
y_tr = np.array(zinc_df.values[:n_tr, 1:-1], dtype=np.float32)
x_tr = zinc_df.values[:n_tr, -1]
y_te = np.array(zinc_df.values[n_tr:, 1:-1], dtype=np.float32)
x_te = zinc_df.values[n_tr:, -1]

# calculate the std of y_tr for loss function
scaler = StandardScaler(copy=False)
y_tr = scaler.fit_transform(y_tr)
y_te = scaler.transform(y_te)

# save the dataset for later use
np.save(y_tr, 'y_tr')
np.save(x_tr, 'x_tr')
np.save(y_te, 'y_te')
np.save(x_te, 'x_te')

# create the language object and map it to strings
lang_obj = lang.Lang(x_tr)
vocab_size = len(lang_obj.idx2ch)
x_tr = lang.preprocessing(x_tr, lang_obj)

# define models
enc_f = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=False)
enc_b = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=True)
attention = nets.BidirectionalAttention(128)
fcuk_props = nets.FullyConnectedUnits([256, 'tanh', 0.25, 256, 'tanh', 0.10, ])
decoder = nets.AttentionDecoder(vocab_size=vocab_size)

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
for epoch in range(500):
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
            ys_hat = fcuk(attention_weights)
            loss += tf.losses.mean_squared_error(ys_hat, ys)
            dec_input = tf.expand_dims([lang.ch2idx['G']] * BATCH_SZ, 1)

            dec_hidden = decoder.initialize_hidden_state()
            for t in range(xs.shape[1]):
                ch_hat, dec_hidden, _ = decoder(dec_input, dec_hidden, attention)
                loss += seq_loss(xs[:, t], ch_hat)
                dec_input = tf.expand_dims(xs[:, t], 1)


        total_loss += loss
        variables = []
        for model in self.models:
            variables += model.variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
        if batch % 10 == 0:
            print("epoch %s batch %s loss %s" % (epoch, batch, np.asscalar(loss.numpy())))
