"""
bayesian.py

Handles the bayesian layers and gaussian mixture models.
"""

# imports
import tensorflow as tf
import numpy as np
try:
    tf.enable_eager_execution()
except:
    pass
import tensorflow_probability as tfp
tfd = tfp.distributions

# utility classes
class MixDistributionLayer(tf.keras.Model):
    """
    MixDistributionLayer takes the input and returns a Gaussian mixture distribution.


    """
    def __init__(self, n_components, dim, probs = None, return_params = False):
        super(MixDistributionLayer, self).__init__()
        self.n_components = n_components
        self.dim = dim
        self.probs = probs
        self.return_params = return_params
        if type(self.probs) == type(None):
            self.probs = np.true_divide(np.ones(self.n_components),
                                        self.n_components)

        self.D = tf.keras.layers.Dense(n_components * dim * 2)

    def __call__(self, x):
        x = self.D(x)
        x = tf.reshape(x, (self.n_components, self.dim, 2))
        locs = x[:, :, 0]
        scales = x[:, :, 1]
        if self.return_params == True:
            return locs, scales
        else:
            mixture_distribution = tfd.Categorical(self.probs)
            components_distribution = tfd.MultivariateNormalDiag(
                                        loc=locs,
                                        scale_diag=scales)
            gm = tfd.MixtureSameFamily(mixture_distribution = mixture_distribution,
                                    components_distribution=components_distribution)
            return gm
