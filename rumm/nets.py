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

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
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
        # enc_output shape == (batch_size, max_length, hidden_size)

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))

class AttentionDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.fc = tf.keras.layers.Dense(vocab_size)

    def __call__(self, attention, x):
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(attention, 1), x], axis=-1)
        x, state =  self.gru(x)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc(x)
        return x

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
        super(BidirectionalWideAttention, self).__init__()
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
