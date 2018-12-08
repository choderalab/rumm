#!/usr/bin/env python

'''
Nets

Handles the structures of neural networks.
'''

# imports
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import logging
import sys



# helper functions
def gru(units):
    # CuDNNGRU is much faster than gru, but is only compatiable when GPU is available.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')

# utility classes
class FullyConnectedUnits(tf.keras.Model):
    """
    Fully Connected Units, consisting of dense layers, dropouts, and activations.

    Parameters
    ----------
    config : a list of str, int, or float.
        A mapping of activation function, layers, and dropouts were constructed.
    """

    def __init__(self, config):
        super(FullyConnectedUnits, self).__init__()
        self.config = config
        self.callables = ['C' + str(idx) for idx in range(len(self.config))]
        self.build_net()
        self.fixed = False

    def build_net(self):
        """
        Construct a FullyConnectedUnits object based on input sequence.
        """
        for idx, value in enumerate(self.config):
            if isinstance(value, str):
                assert (value in ['tanh', 'relu', 'sigmoid']), "Can't identify activation function."
                if value == 'tanh':
                    setattr(self, 'C' + str(idx), tf.tanh)
                elif value == 'relu':
                    setattr(self, 'C' + str(idx), tf.nn.relu)
                elif value == 'sigmoid':
                    setattr(self, 'C' + str(idx), tf.sigmoid)

            elif isinstance(value, int):
                assert (value >= 1), "Can\'t have fewer than one neuron."
                setattr(self, 'C' + str(idx), tf.keras.layers.Dense(value))

            elif isinstance(value, float):
                assert (value < 1), "Can\'t have dropouts larger than one."
                setattr(self, 'C' + str(idx), lambda x: tf.layers.dropout(x, rate=value))

    def __call__(self, x_tensor):
        for callable in self.callables:
            x_tensor = getattr(self, callable)(x_tensor)

        return x_tensor

    def initialize(self, x_shape):
        self.__call__(tf.zeros(x_shape))

    @property
    def input_shape(self):
        return self.get_input_shape_at(0)


class Encoder(tf.keras.Model):
    """
    Encoder encodes a preprocessed sequence data using GRU.

    Parameters
    ----------
    batch_sz : int, batch size
    enc_units : int, the number of neurons in encoder
    embedding_dim : the dimmension of space to which the sequence data is projected
    reverse : Boolean. If True, the sequence is reversed.
        This is useful for bidirectional RNN.
    """

    def __init__(self, vocab_size, embedding_dim=16, enc_units=128, batch_sz=16, reverse = False):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru_f = gru(self.enc_units)
        self.reverse = reverse

    def __call__(self, x_tensor):
        x_tensor = self.embedding(x_tensor)
        if self.reverse == True:
            x_tensor = tf.reverse(x_tensor, [1])
        output, state = self.gru_f(x_tensor, initial_state = tf.zeros((self.batch_sz, self.enc_units)))
        # output, state = self.gru_f(x_tensor)
        return output, state

    def initialize(self, x_shape):
        self.__call__(tf.zeros(x_shape))

class BidirectionalAttention(tf.keras.Model):
    """
    Attention constructs the attention weights in seq2seq model.

    """
    def __init__(self, units):
        super(BidirectionalAttention, self).__init__()
        self.units = units
        self.W1_f = tf.keras.layers.Dense(self.units)
        self.W1_b = tf.keras.layers.Dense(self.units)
        self.W2_f = tf.keras.layers.Dense(self.units)
        self.W2_b = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, eo_f, eo_b, h_f, h_b):
        ht_f = tf.expand_dims(h_f, 1)
        ht_b = tf.expand_dims(h_b, 1)
        score = tf.nn.tanh(self.W1_f(eo_f) + self.W1_b(eo_b) + self.W2_f(ht_f) + self.W2_b(ht_b))
        attention_weights = tf.layers.flatten(tf.nn.softmax(self.V(score), axis=1))
        return attention_weights

    @property
    def input_shapes(self):
        return [model.get_input_shape_at(0) for model in [self.W1_f, self.W1_b, self.W2_f, self.W2_b, self.V]]

    def initialize(self, eo_shape, h_shape):
        self.__call__(tf.zeros(eo_shape), tf.zeros(eo_shape), tf.zeros(h_shape), tf.zeros(h_shape))

class BidirectionalWideAttention(tf.keras.Model):
    """
    Attention constructs the attention weights in seq2seq model.

    """
    def __init__(self, units):
        super(BidirectionalAttention, self).__init__()
        self.units = units
        self.W1_f = tf.keras.layers.Dense(self.units)
        self.W1_b = tf.keras.layers.Dense(self.units)
        self.W2_f = tf.keras.layers.Dense(self.units)
        self.W2_b = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, eo_f, eo_b, h_f, h_b):
        ht_f = tf.expand_dims(h_f, 1)
        ht_b = tf.expand_dims(h_b, 1)
        score_f = tf.nn.tanh(self.W1_f(eo_f) + self.W1_b(ht_f))
        score_b = tf.nn.tanh(self.W2_f(eo_b) + self.W2_b(ht_b))
        score = tf.concat([score_f, score_b], axis=-1)
        attention_weights = tf.layers.flatten(tf.nn.softmax(self.V(score), axis=1))
        return attention_weights

    @property
    def input_shapes(self):
        return [model.get_input_shape_at(0) for model in [self.W1_f, self.W1_b, self.W2_f, self.W2_b, self.V]]

    def initialize(self, eo_shape, h_shape):
        self.__call__(tf.zeros(eo_shape), tf.zeros(eo_shape), tf.zeros(h_shape), tf.zeros(h_shape))


class Box:
    """
    The wrapper of all layers in the model.

    Parameters
    ----------
    models : a nested structure of models
    n_epochs : int, number of epochs
    batch_sz : int, batch size
    loss_fn : a function that describe the loss during training
    """

    def __init__(self, flow, models, n_epochs=10, batch_sz=1, loss_fn=tf.losses.mean_squared_error):
        self.flow = flow
        self.models = models
        self.n_epochs = n_epochs
        self.batch_sz = batch_sz
        self.loss_fn = loss_fn
        for model in models:
            if hasattr(model, 'batch_sz'):
                model.batch_sz = self.batch_sz

    def train(self, x_tr, y_tr,
                optimizer=tf.train.AdamOptimizer(),
                loss_fn=tf.losses.mean_squared_error):
        """
        Train the model with training set, x and y

        Parameters
        ----------
        flow : the function which takes the data and the model, to make a prediction
        x_tr : np.ndarry
        y_tr : np.ndarry, has to match the number of samples in x_tr
        """
        # convert them into tensors
        y_tr = np.array(y_tr, dtype=np.float32)
        x_tr = tf.convert_to_tensor(x_tr)

        if y_tr.ndims == 1 or y_tr.shape[1] == 1:
            y_tr = np.transpose([y_tr.flatten()])
        y_tr = tf.convert_to_tensor(y_tr)
        # make them into a dataset object
        ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(y_tr.shape[0])
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_sz))

        # loop through the epochs
        for epoch in range(self.n_epochs):
            total_loss = 0 # initialize the total loss at the beginning to be 0

            # loop through the batches
            for (batch, (xs, ys)) in enumerate(ds):

                # the loss at the beginning of the batch is zero
                loss = 0

                with tf.GradientTape() as tape: # for descent
                    ys_hat = self.flow(xs, self.models) # the flow function takes xs and models to make prediction
                    loss += self.loss_fn(ys_hat, ys)
                total_loss += loss
                variables = []
                for model in self.models:
                    variables += model.variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
                if batch % 10 == 0:
                    print("epoch %s batch %s loss %s" % (epoch, batch, np.asscalar(loss.numpy())))

    def predict(self, x_te):
        """
        Make predictions on the x of test set.

        Parameters
        ----------
        flow : the function that takes data and the models and output ys
        x_te : np.ndarry, the test data
        """

        ys_hat_all = np.array([])
        x_te = tf.convert_to_tensor(x_te)
        ds_te = tf.data.Dataset.from_tensor_slices((x_te))
        ds_te = ds.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_sz))
        for xs in ds_te:
            ys_hat = self.flow(xs, self.models)
            ys_hat_all = np.concatenate([ys_hat_all, ys_hat.numpy().flatten()], axis=0)
        return ys_hat_all


    def save_weights(self, file_path):
        """
        Save the model. Note that it is necessary to also save the shape of the input.
        """
        import os
        os.system('rm -rf ' + file_path)
        os.system('mkdir ' + file_path)
        for idx, model in enumerate(self.models):
            model.save_weights('%s/%s.h5' % (file_path, idx))
            # model.save_weights('%s/%s.h5' % (file_path, idx))

    def load_weights(self, file_path):
        """
        Restore the model.
        """
        for idx, model in self.models:
            model.load_weights('%s/%s.h5' % (file_path, idx))
