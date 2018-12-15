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
BATCH_SZ = 128

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
pickle.dump(scaler, open('scaler.p', 'wb'))

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
enc_f = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=False)
enc_b = nets.Encoder(vocab_size=vocab_size, batch_sz = BATCH_SZ, reverse=True)
attention = nets.BidirectionalWideAttention(128)
fcuk = nets.FullyConnectedUnits([512, 'tanh', 0.30, 512, 'tanh', 0.30, 512, 'tanh', 0.25])
fcuk_props = nets.FullyConnectedUnits([9])
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


loss_scale_0 = 0.0
loss_scale_1 = 0.0

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
            ys_hat = fcuk_props(attention_weights)
            loss0 = tf.losses.mean_squared_error(ys_hat, ys)

            if loss_scale_0 == 0.0:
                loss_scale_0 = loss0

            loss += tf.div(loss0, loss_scale_0)

            dec_input = tf.expand_dims([lang_obj.ch2idx['G']] * BATCH_SZ, 1)
            dec_hidden = decoder.initialize_hidden_state()
            loss1 = 0
            for t in range(xs.shape[1]):
                ch_hat, dec_hidden = decoder(dec_input, dec_hidden, attention_weights)
                loss1 += seq_loss(xs[:, t], ch_hat)
                dec_input = tf.expand_dims(xs[:, t], 1)
            if loss_scale_1 == 0.0:
                loss_scale_1 = loss1
            loss += tf.div(loss1, loss_scale_1)

        total_loss += loss
        variables = enc_f.variables + enc_b.variables + attention.variables + fcuk_props.variables + decoder.variables + fcuk.variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
        if batch % 10 == 0:
            print("epoch %s batch %s loss %s" % (epoch, batch, np.asscalar(loss.numpy())))

        fcuk.save_weights('./fcuk.h5')
        enc_f.save_weights('./enc_f.h5')
        enc_b.save_weights('./enc_b.h5')
        attention.save_weights('./attention_weights.h5')
        fcuk_props.save_weights('./fcuk_props.h5')
        decoder.save_weights('./decoder.h5')
