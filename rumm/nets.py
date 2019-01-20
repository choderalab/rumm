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
    # CuDNNGRU is much faster than gru, but is only compatiable when GPU is available.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                                    recurrent_regularizer=tf.keras.regularizers.l1(l=0.001),
                                    bias_regularizer=tf.keras.regularizers.l2(l=0.01),
                                    go_backwards=reverse)
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
        self.fixed = False
        self.dropout_list = []
        self.build_net()

    def build_net(self):
        """
        Construct a FullyConnectedUnits object based on input sequence.
        """
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
                setattr(self, 'C' + str(idx), lambda x: tf.nn.dropout(x, value))
                self.dropout_list.append('C' + str(idx))


    @tf.contrib.eager.defun
    def __call__(self, x_tensor):
        for callable in self.callables:
            x_tensor = getattr(self, callable)(x_tensor)
        return x_tensor

    def initialize(self, x_shape):
        self.__call__(tf.zeros(x_shape))

    def switch_to_test(self):
        for callable in self.dropout_list:
            setattr(self, callable, lambda x: x)


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
        if reverse == False:
            self.gru_f = gru(self.enc_units)
        elif reverse == True:
            self.gru_f = gru(self.enc_units, True)

    def __call__(self, x_tensor):
        x_tensor = self.embedding(x_tensor)
        output, state = self.gru_f(x_tensor, initial_state = tf.zeros((self.batch_sz, self.enc_units)))
        return output, state

    def initialize(self, x_shape):
        self.__call__(tf.zeros(x_shape))

class Decoder(tf.keras.Model):
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
        return tf.zeros((self.batch_sz, self.dec_units))

class AttentionDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=16, dec_units=128, batch_sz=16):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.gru = gru(self.dec_units)

    @tf.contrib.eager.defun
    def __call__(self, x, hidden, attention_weights):
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(attention_weights, 1), x], axis=-1)
        x, state =  self.gru(x)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc(x)
        return x, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))

class DeepAttentionDecoder(tf.keras.Model):
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

    @tf.contrib.eager.defun
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
        return tf.zeros((self.batch_sz, self.dec_units))



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

    @tf.contrib.eager.defun
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

    @tf.contrib.eager.defun
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


class OneHotDecoder(tf.keras.Model):
    def __init__(self, vocab_size, dec_units = 32, batch_sz = 128, max_len = 64):
        super(OneHotDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.dec_units = dec_units
        self.batch_sz = batch_sz
        self.gru = nets.gru(self.dec_units)
        self.gru1 = nets.gru(self.dec_units)
        self.max_len = max_len
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size))
        self.D = tf.keras.layers.Dense(dec_units)

    def __call__(self, x):
        x = self.D(x)
        x = tf.expand_dims(x, 1)
        x = tf.tile(x, [1, self.max_len, 1])
        x_0, hidden_0 = self.gru(x)
        x_1, hidden_1 = self.gru1(x_0)
        x = tf.concat([x_0, x_1], -1)
        x = self.fc(x)
        return x
