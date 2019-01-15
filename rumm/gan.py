"""
gan.py

Handles the generative adversary network layers.
"""

# =======
# imports
# =======
import tensorflow as tf
import numpy as np
try:
    tf.enable_eager_execution()
except:
    pass
import tensorflow_probability as tfp
tfd = tfp.distributions
import nets

# ===============
# utility classes
# ===============
class GAN(tf.keras.Model):
    """
    Generative adversary network.

    Generator: Takes an input from the source latent space, sample from N(0, 1),
    and concatenate together, to feed into a neural network, to produce a target
    point on latent space.

    Discriminator: Discriminate whether the molecule pair is generic or synthesized.
    """
    def __init__(self,
                 g_config = [128, 'leaky_relu', 128, 'leaky_relu', 128, 'leaky_relu', 8],
                 d_config = [64, 0.10, 'leaky_relu', 64, 0.10, 'leaky_relu', 1, 'sigmoid'],
                 batch_sz = 2048, n_epochs = 500):

        super(GAN, self).__init__()
        self.G = nets.FullyConnectedUnits(g_config)
        self.D = nets.FullyConnectedUnits(d_config)
        self.batch_sz = batch_sz
        self.ds = None

    def g_sample(self, x):
        """
        Sample from N(0, 1), concatenate with the point on the latent space,
        and transform it using the generator.

        """
        r = tf.clip_by_norm(tf.random_normal(x.shape), 1e5)
        x = tf.concatenate([x, r], axis=1)
        y_r = self.G(r)
        return y_r

    def d_predict(self, x, y):
        """
        Predict whether the pair is generic or synthesized.

        Parameters
        ----------
        x : the source molecules
        y : the target molecules

        Returns
        -------
        z : the label of the molecule pairs of being whether generic (1) or
            artificial (0).
        """
        pair = tf.concatenate([x, y], axis=1)
        z = self.D(pair)
        return z

    def load_dataset(self, x, y):
        """
        Load dataset into training.

        Parameters
        ----------
        x : tf.Tensor the source molecules projected on the latent space.
        y : tf.Tensor the target molecules projected on the latent space.

        """
        ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr)).shuffle(y_tr.shape[0])
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_sz))
        self.ds = ds

    def train(self):
        """
        Train the network.

        """
        optimizer=tf.train.AdamOptimizer() # use adam optimizer here
        for epoch in range(self.n_epochs): # loop through epochs
            for (batch, (xs, ys)) in enumerate(self.ds): # loop through datasets
                with tf.GradientTape() as tape:
                    y_r = g_sample(xs)
                    z_generic = self.d_predict(xs, ys)
                    z_synthesized = self.d_predict(xs, y_r)
                    loss_d = -tf.reduce_mean(tf.log(z_generic) + tf.log(1 - z_synthesized))
                    loss_g = tf.reduce_mean(tf.log(1 - z_synthesized))

                # clip the loss function
                loss_d = tf.clip_by_norm(loss_d, 1e5)
                loss_g = tf.clip_by_norm(loss_d, 1e5)

                # get the variables
                var_d = self.D.variables
                var_g = self.G.variables

                # get the gradients, again, with clip
                grad_d = tf.clip_by_norm(tape.gradient(loss_d, var_d), 1e5)
                grad_g = tf.clip_by_norm(tape.gradient(loss_g, var_g), 1e5)

                optimizer.apply_gradients(zip(var_d, grad_d), tf.train.get_or_create_global_step())
                optimizer.apply_gradients(zip(var_g, grad_g), tf.train.get_or_create_global_step())