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
def gru(units, reverse = False):
    """

    Parameters
    ----------
    units :
        param reverse:  (Default value = False)
    reverse :
         (Default value = False)

    Returns
    -------

    """
    # CuDNNGRU is much faster than gru, but is only compatiable when GPU is available.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform',
                                    kernel_regularizer=tf.keras.regularizers.l1(l=0.0001),
                                    recurrent_regularizer=tf.keras.regularizers.l1(l=0.00001),
                                    bias_regularizer=tf.keras.regularizers.l1(l=0.0001),
                                    go_backwards=reverse)
    else:
        return tf.keras.layers.GRU(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')

def lstm(units):
    """

    Parameters
    ----------
    units :


    Returns
    -------

    """
    # CuDNNGRU is much faster than gru, but is only compatiable when GPU is available.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_normal',
                                    kernel_regularizer=tf.keras.regularizers.l1(l=0.0001),
                                    recurrent_regularizer=tf.keras.regularizers.l1(l=0.0001),
                                    bias_regularizer=tf.keras.regularizers.l1(l=0.0001))
    else:
        return tf.keras.layers.LSTM(units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_activation='sigmoid',
                               recurrent_initializer='glorot_uniform')


class OneHotShallowDecoder(tf.keras.Model):
    """ """
    def __init__(self, vocab_size, dec_units = 32, batch_sz = 128, max_len = 64):
        super(OneHotShallowDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.lstm = lstm(self.dec_units)
        self.lstm0 = lstm(self.dec_units)
        self.max_len = max_len
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
        self.D = tf.keras.layers.Dense(2 * dec_units)
        self.D1 = tf.keras.layers.Dense(2 * dec_units)


    def __call__(self, x):
        x = self.D(x)
        x = self.D1(x)
        x = tf.expand_dims(x, 1)
        x = tf.tile(x, [1, self.max_len, 1])
        x, _, _ = self.lstm(x)
        x, _, _ = self.lstm0(x)
        x_ = self.fc(x)
        return x_

class OneHotDecoder(tf.keras.Model):
    def __init__(self, vocab_size, dec_units = 32, batch_sz = 128, max_len = 64):
        super(OneHotDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.gru_f = gru(self.dec_units)
        self.gru_b = gru(self.dec_units, reverse=True)
        self.lstm1_f = lstm(self.dec_units)
        self.lstm1_b = lstm(self.dec_units, reverse=True)
        self.lstm2 = lstm(self.dec_units)
        self.lstm3 = lstm(self.dec_units)

        self.max_len = max_len
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
        self.fc1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
        self.fc2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
        self.D = tf.keras.layers.Dense(2 * dec_units)
        self.D1 = tf.keras.layers.Dense(2 * dec_units)
        # self.D_bypass0 = tf.keras.layers.Dense(2 * dec_units)
        # self.D_bypass = tf.keras.layers.Dense(vocab_size * max_len)

    @tf.contrib.eager.defun
    def _call(self, x_):
        x = self.D(x_)
        x = self.D1(x)
        x = tf.expand_dims(x, 1)
        x = tf.tile(x, [1, self.max_len, 1])
        x = self.fc1(x)
        x = self.fc2(x)
        x_f, _, _ = self.lstm1_f(x)
        x_b, _, _ = self.lstm1_b(x)
        x = tf.concat([x_f, x_b], -1)
        x, _, _ = self.lstm2(x)
        x, _, _ = self.lstm3(x)
        x = self.fc(x)
        # x = self.fc1(x)

        # x_bypass = self.D_bypass(self.D_bypass0(x_))
        # x = self.fc(x) + tf.reshape(x_bypass, [x_bypass.shape[0], self.max_len, self.vocab_size])
        return x

    def __call__(self, x):
        return self._call(x)

class ConvEncoder(tf.keras.Model):
    """ Encoding sequence onto a latent space.

    Attributes
    ----------
    conv_units : list
        list of units of conv layers
    pool_sizes : list
        list of sizes of pooling layers
    conv_kernel_sizes : list
        list of sizes of conv kernels
    fcs : list
        list of fully connected layers configurations.
            - str -> activation function
            - int -> one layer with neuron
            - float -> dropout

    Methods
    -------
    build : building the model
    switch : switch between training and test

    """

    def __init__(
        self,
        conv_units: list,
        # pool_sizes: list,
        conv_kernel_sizes: list,
        fcs: list) -> None:

        # bookkeeping
        super(ConvEncoder, self).__init__()
        self.conv_units = conv_units
        # self.pool_sizes = pool_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.fcs = fcs

        # define the workflow
        self.workflow = []

        # run build() once to construct the model
        self.build()


    def build(self):
        # assert len(self.conv_units) == len(self.pool_sizes)
        assert len(self.conv_units) == len(self.conv_kernel_sizes)

        # set convolution layers
        for idx in range(len(self.conv_units)):
            # get the names
            conv_name = 'C' + str(idx)
            # pool_name = 'P' + str(idx)
            self.workflow.append(conv_name)
            # self.workflow.append(pool_name)

            # get the configs
            conv_unit = self.conv_units[idx]
            conv_kernel_size = self.conv_kernel_sizes[idx]
            # pool_size = self.pool_sizes[idx]

            # set the attributes
            # conv layers
            setattr(
                self,
                conv_name,
                tf.keras.layers.Conv1D(
                    conv_unit, conv_kernel_size,
                    kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                    bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                    activation = tf.nn.tanh))

            '''
            # pool layers
            setattr(
                self,
                pool_name,
                tf.keras.layers.MaxPooling1D(pool_size)
                )
            '''


        # flatten it
        setattr(self, 'F', tf.layers.flatten)
        self.workflow.append('F')

        for idx, value in enumerate(self.fcs):
            # str -> activation function
            if isinstance(value, str):
                # within our selction
                assert value in ["tanh", "relu", "sigmoid", "leaky_relu", "elu"]

                name = 'A' + str(idx)
                if value == 'tanh':
                    setattr(self, name, tf.tanh)
                elif value == 'relu':
                    setattr(self, name, tf.nn.relu)
                elif value == 'sigmoid':
                    setattr(self, name, tf.sigmoid)
                elif value == 'leaky_relu':
                    setattr(self, name, tf.nn.leaky_relu)
                elif value == 'elu':
                    setattr(self, name, tf.nn.elu)

            # int -> fc layer
            elif isinstance(value, int):
                assert value >= 1
                name = 'D' + str(idx)
                setattr(self, name,
                    tf.keras.layers.Dense(
                        value,
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                        bias_regularizer=tf.keras.regularizers.l2(0.01)))

            # float -> dropout
            elif isinstance(value, float):
                assert value < 1
                name = 'O' + str(idx)
                setattr(self, name,
                    tf.layers.Dropout(value))

            self.workflow.append(name)


    @tf.contrib.eager.defun
    def _call(self, x):
        for name in self.workflow:
            x = getattr(self, name)(x)
        return x


    def __call__(self, x):
        return self._call(x)

    def switch(self, to_test = True):
        if to_test == True:
            for idx, name in enumerate(self.workflow):
                if name.startswith('O'):
                    setattr(self, name, lambda x: x)
        else:
            for idx, value in enumerate(self.fcs):
                if isinstance(value, float):
                    assert value < 1
                    name = 'O' + str(idx)
                    setattr(self, name,
                       tf.layers.Dropout(value))

# to be merged with OneHotDecoder
class SimpleDecoder(tf.keras.Model):
    def __init__(self, vocab_size, dec_units = 32, batch_sz = 128, max_len = 64):
        super(SimpleDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.max_len = max_len
        self.D = tf.keras.layers.Dense(dec_units)
        self.D0 = tf.keras.layers.Dense(dec_units)
        self.D1 = tf.keras.layers.Dense(self.vocab_size * self.max_len)

    def __call__(self, x):
        x = self.D(x)
        x = self.D0(x)
        x = self.D1(x)
        x = tf.reshape(x, [x.shape[0], self.max_len, self.vocab_size])
        return x


# utility classes
class FullyConnectedUnits(tf.keras.Model):
    """Fully Connected Units, consisting of dense layers, dropouts, and activations."""

    def __init__(self, config):
        super(FullyConnectedUnits, self).__init__()
        self.config = config
        self.callables = ['C' + str(idx) for idx in range(len(self.config))]
        self.fixed = False
        self.dropout_list = []
        self.build_net()

    def build_net(self):
        """Construct a FullyConnectedUnits object based on input sequence."""
        for idx, value in enumerate(self.config):
            if isinstance(value, str):
                assert (value in ['tanh', 'relu', 'sigmoid', 'leaky_relu']), "Can't identify activation function."
                if value == 'tanh':
                    setattr(self, 'C' + str(idx), tf.tanh)
                elif value == 'relu':
                    setattr(self, 'C' + str(idx), tf.nn.relu)
                elif value == 'sigmoid':
                    setattr(self, 'C' + str(idx), tf.sigmoid)
                elif value == 'leaky_relu':
                    setattr(self, 'C' + str(idx), tf.nn.leaky_relu)

            elif isinstance(value, int):
                assert (value >= 1), "Can\'t have fewer than one neuron."
                setattr(self, 'C' + str(idx), tf.keras.layers.Dense(value,
                  kernel_regularizer = tf.keras.regularizers.l2(0.01),
                  bias_regularizer = tf.keras.regularizers.l2(0.01)))

            elif isinstance(value, float):
                assert (value < 1), "Can\'t have dropouts larger than one."
                setattr(self, 'C' + str(idx), tf.layers.Dropout(value))
                self.dropout_list.append('C' + str(idx))

    @tf.contrib.eager.defun
    def _call(self, x_tensor):
        for callable in self.callables:
            x_tensor = getattr(self, callable)(x_tensor)
        return x_tensor

    def __call__(self, x):
        return self._call(x)

    def initialize(self, x_shape):
        """

        Parameters
        ----------
        x_shape :


        Returns
        -------

        """
        self.__call__(tf.zeros(x_shape))

    def switch_to_test(self):
        """ """
        for callable in self.dropout_list:
            setattr(self, callable, lambda x: x)


    @property
    def input_shape(self):
        """ """
        return self.get_input_shape_at(0)


class GRUEncoder(tf.keras.Model):
    """Encoder encodes a preprocessed sequence data using GRU."""

    def __init__(self, vocab_size, enc_units=128, batch_sz=16, reverse = False):
        super(GRUEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        if reverse == False:
            self.gru_f = gru(self.enc_units)
        elif reverse == True:
            self.gru_f = gru(self.enc_units, True)

        self.gru1 = gru(self.enc_units)
        self.gru2 = gru(self.enc_units)

    @tf.contrib.eager.defun
    def _call(self, xs):
        x = tf.one_hot(xs, 36)
        x, state0 = self.gru_f(x)
        x, state1 = self.gru1(x)
        x, state2 = self.gru2(x)
        state = tf.concat([state0, state1, state2], axis=-1)
        return x, state

    def __call__(self, x):
        return self._call(x)

class Decoder(tf.keras.Model):
    """ """
    def __init__(self, vocab_size, embedding_dim=16, dec_units=128, batch_sz=16):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, x, hidden, enc_output):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

    def initialize_hidden_state(self):
        """ """
        return tf.zeros((self.batch_sz, self.dec_units))

class AttentionDecoder(tf.keras.Model):
    """ """
    def __init__(self, vocab_size, embedding_dim=16, dec_units=128, batch_sz=16):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.gru = gru(self.dec_units)

    # @tf.contrib.eager.defun
    def __call__(self, x, hidden, attention_weights):
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(attention_weights, 1), x], axis=-1)
        x, state =  self.gru(x)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc(x)
        return x, state

    def initialize_hidden_state(self):
        """ """
        return tf.zeros((self.batch_sz, self.dec_units))

class DeepAttentionDecoder(tf.keras.Model):
    """ """
    def __init__(self, vocab_size, embedding_dim=16, dec_units=128, batch_sz = 128):
        super(DeepAttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dec_units = dec_units
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.gru = gru(self.dec_units)
        self.D0 = tf.keras.layers.Dense(dec_units)
        self.D1 = tf.keras.layers.Dense(dec_units)
        self.D2 = tf.keras.layers.Dense(dec_units)
        self.D3 = tf.keras.layers.Dense(dec_units)
        self.batch_sz = batch_sz

    # @tf.contrib.eager.defun
    def __call__(self, x, attention_weights, hidden):
        attention_weights = tf.nn.leaky_relu(self.D0(attention_weights))
        attention_weights = tf.nn.leaky_relu(self.D1(attention_weights))
        hidden = tf.nn.leaky_relu(self.D2(hidden))
        context = tf.concat([attention_weights, hidden], axis=-1)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1),
                       x], axis=-1)
        x = self.D3(x)

        x, hidden = self.gru(x)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc(x)
        return x, hidden

    def initialize_hidden_state(self):
        """ """
        return tf.zeros((self.batch_sz, self.dec_units))



class BidirectionalAttention(tf.keras.Model):
    """Attention constructs the attention weights in seq2seq model."""
    def __init__(self, units):
        super(BidirectionalAttention, self).__init__()
        self.units = units
        self.W1_f = tf.keras.layers.Dense(self.units)
        self.W1_b = tf.keras.layers.Dense(self.units)
        self.W2_f = tf.keras.layers.Dense(self.units)
        self.W2_b = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    # @tf.contrib.eager.defun
    def __call__(self, eo_f, eo_b, h_f, h_b):
        ht_f = tf.expand_dims(h_f, 1)
        ht_b = tf.expand_dims(h_b, 1)
        score = tf.nn.tanh(self.W1_f(eo_f) + self.W1_b(eo_b) + self.W2_f(ht_f) + self.W2_b(ht_b))
        attention_weights = tf.layers.flatten(tf.nn.softmax(self.V(score), axis=1))
        return attention_weights

    @property
    def input_shapes(self):
        """ """
        return [model.get_input_shape_at(0) for model in [self.W1_f, self.W1_b, self.W2_f, self.W2_b, self.V]]

    def initialize(self, eo_shape, h_shape):
        """

        Parameters
        ----------
        eo_shape :
            param h_shape:
        h_shape :


        Returns
        -------

        """
        self.__call__(tf.zeros(eo_shape), tf.zeros(eo_shape), tf.zeros(h_shape), tf.zeros(h_shape))

class BidirectionalWideAttention(tf.keras.Model):
    """Attention constructs the attention weights in seq2seq model."""
    def __init__(self, units):
        super(BidirectionalWideAttention, self).__init__()
        self.units = units
        self.W1_f = tf.keras.layers.Dense(self.units)
        self.W1_b = tf.keras.layers.Dense(self.units)
        self.W2_f = tf.keras.layers.Dense(self.units)
        self.W2_b = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    # @tf.contrib.eager.defun
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
        """ """
        return [model.get_input_shape_at(0) for model in [self.W1_f, self.W1_b, self.W2_f, self.W2_b, self.V]]

    def initialize(self, eo_shape, h_shape):
        """

        Parameters
        ----------
        eo_shape :
            param h_shape:
        h_shape :


        Returns
        -------

        """
        self.__call__(tf.zeros(eo_shape), tf.zeros(eo_shape), tf.zeros(h_shape), tf.zeros(h_shape))


class OneHotDecoder(tf.keras.Model):
    def __init__(self, vocab_size, dec_units = 32, batch_sz = 128, max_len = 64):
        super(OneHotDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.lstm = lstm(self.dec_units)
        self.max_len = max_len
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
        self.D = tf.keras.layers.Dense(2 * dec_units)
        self.D1 = tf.keras.layers.Dense(2 * dec_units)


    def __call__(self, x):
        x = self.D(x)
        x = self.D1(x)
        x = tf.expand_dims(x, 1)
        x = tf.tile(x, [1, self.max_len, 1])
        x, _, _ = self.lstm(x)
        x_ = self.fc(x)
        return x_
