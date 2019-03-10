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
class ConditionalGAN(tf.keras.Model):
    """Generative adversary network.

    Generator: Takes an input from the source latent space, sample from N(0, 1),
    and concatenate together, to feed into a neural network, to produce a target
    point on latent space.

    Discriminator: Discriminate whether the molecule pair is generic or synthesized.

    Parameters
    ----------

    Returns
    -------

    """
    def __init__(self,
                 g_config = [512, 'leaky_relu', 1024, 'leaky_relu', 1024,
                   'leaky_relu', 256],
                 d_config = [128, 0.25, 'leaky_relu', 128, 0.25, 'leaky_relu', 1, 'sigmoid'],
                 batch_sz = 2048, n_epochs = 500):

        super(ConditionalGAN, self).__init__()
        self.G = nets.FullyConnectedUnits(g_config)
        self.D = nets.FullyConnectedUnits(d_config)
        self.batch_sz = batch_sz
        self.ds = None
        self.n_epochs = n_epochs

    # @tf.contrib.eager.defun
    def g_sample(self, x):
        """Sample from N(0, 1), concatenate with the point on the latent space,
        and transform it using the generator.

        Parameters
        ----------
        x :


        Returns
        -------

        """
        r = tf.clip_by_norm(tf.random_normal(x.shape,
            dtype=tf.float32), 1e5)
        x = tf.concat([x, r], axis=1)
        y_r = self.G(x)
        return y_r

    # @tf.contrib.eager.defun
    def d_predict(self, x, y):
        """Predict whether the pair is generic or synthesized.

        Parameters
        ----------
        x :
            type x: the source molecules
        y :
            type y: the target molecules

        Returns
        -------

        """
        pair = tf.concat([x, y], axis=0)
        z = self.D(pair)
        return z

    def load_dataset(self, x, y):
        """Load dataset into training.

        Parameters
        ----------
        x :
            param y:
        y :


        Returns
        -------

        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((x, y))# .shuffle(y.shape[0])
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_sz))
        self.ds = ds

    def train(self):
        """Train the network."""
        optimizer=tf.train.AdamOptimizer() # use adam optimizer here
        for epoch in range(self.n_epochs): # loop through epochs
            for (batch, (xs, ys)) in enumerate(self.ds): # loop through datasets
                with tf.GradientTape(persistent=True) as tape:

                    y_r = self.g_sample(xs)
                    z_generic = self.d_predict(xs, ys)
                    z_synthesized = self.d_predict(xs, y_r)
                    loss_d = -tf.reduce_mean(tf.log(z_generic) + tf.log(1 - z_synthesized))
                    loss_g = tf.reduce_mean(tf.log(1 - z_synthesized))

                    loss_d = tf.clip_by_norm(loss_d, 1e5)
                    loss_g = tf.clip_by_norm(loss_d, 1e5)

                # get the variables
                var_d = self.D.variables
                var_g = self.G.variables

                # get the gradients, again, with clip
                grad_d = tape.gradient(loss_d, var_d)
                grad_g = tape.gradient(loss_g, var_g)

                optimizer.apply_gradients(zip(grad_d, var_d), tf.train.get_or_create_global_step())
                optimizer.apply_gradients(zip(grad_g, var_g), tf.train.get_or_create_global_step())

            self.D.save_weights('./D.h5')
            self.G.save_weights('./G.h5')
            print('D: %s' % loss_d.numpy())
            print('G: %s' % loss_g.numpy())
