"""
vmf.py

Handles the sampling of vmf distribution.
"""

import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_probability as tfp
tfd = tfp.distributions

@tf.contrib.eager.defun
def sample_v(mu, d, batch_sz):
    """ Sample a vector vertical to mu on the spherical plane.

    Parameters
    ----------
    mu : tf.Tensor, shape=(batch_sz, d)
        unit vector in the space
    d : int or integer tensor
        dimension
    batch_sz : int or int tensor

    Returns
    -------
    v : tf.Tensor, shape=(batch_sz, d)
        random tensor prependicular to mu
    """
    # sample a random vector
    # (batch_sz, d)
    v = tf.random.normal([batch_sz, d], dtype=tf.float32)
    # projection
    # (batch_sz, d)

    proj_mu_v = tf.multiply(
        mu, # (batch_sz, d)
        tf.expand_dims(tf.reduce_sum(
            tf.multiply(v, mu), 1), 1) *\
        tf.ones((1, d), dtype=tf.float32))

    # subtract the projection to get the orthogonal part
    # and normalize
    # (batch_sz, d)
    v = v - proj_mu_v
    v = tf.linalg.l2_normalize(v, axis=1)
    return v


@tf.contrib.eager.defun
def sample_w(kappa, d, batch_sz):
    """ Rejection sampling for weights.

    Parameters
    ----------
    kappa : tf.Tensor, shape=(1,)
        Hyperparameter.
    d : int or integer tensor
        Dimension.
    batch_sz : int or integer tensor
        Batch size.

    $$
    b = \frac{d}{\sqrt{4 * \kappa ^ 2 + d ^ 2} + 2 * \kappa}\\
    x = \frac{1 - b}{1 + b}
    c = \kappa * x + d * \log(1 - x ^ 2)

    $$

    Sample z ~ Beta(d/2, d/2).
    Sample u ~ Uniform(0, 1).

    $$
    w = \frac{1 - (1 + b) * z}{1 - (1 - b) * z}
    $$

    Return $w$ if:

    $$
    \kappa * w + d * \log(1 - x * w) - c >= \log(u)
    $$
    """
    # =============
    # deterministic
    # =============
    d = d - 1 # on spherical space
    b = tf.div(
        d * 1.0,
        tf.sqrt(4. * kappa ** 2 + d ** 2) + 2 * kappa)
    x = tf.div(
        1. - b,
        1. + b)
    c = kappa * x\
        + d * tf.log(1 - x ** 2)

    # ==================
    # rejection sampling
    # ==================

    # specify the distributions
    # (batch_sz,)
    z_rv = tfd.Beta(d/2., d/2.)
    u_rv = tfd.Uniform()

    # sample scheme
    # TODO: profile this and try making it faster
    res = tf.zeros([batch_sz])
    @tf.contrib.eager.defun
    def loop_body(res):
        # sample z and u
        z = z_rv.sample(batch_sz)
        u = u_rv.sample(batch_sz)
        w = tf.div(
            1. - (1. + b) * z,
            1. - (1. - b) * z)

        # compute masks
        # assign value only where:
        #   - acceptance criteria is satisfied
        #   - res is not zero
        accept_mask = tf.greater_equal(
                kappa * w\
                + d * tf.log(1. - x * w)\
                - c,
            tf.log(u))
        non_zero_mask = tf.equal(res, tf.constant(0., dtype=tf.float32))
        mask = tf.logical_and(accept_mask, non_zero_mask)

        # update the result where the mask shows True
        res = tf.where(mask, w, res)
        return res

    res = tf.while_loop(
        lambda x: tf.logical_not(tf.reduce_all(
            tf.greater(x, tf.constant(0, dtype=tf.float32)))), # condition
        lambda x: loop_body(x), # body
        [res]) # loop vars

    return res

@tf.contrib.eager.defun
def sample_z(mu, kappa, d, batch_sz):
    """ Perturb z on the spherical space.

    Parameters
    ----------
    mu : tf.Tensor
        vector to perturb
    kappa
        constant
    d : int or integer tensor
        dimension
    batch_sz : int or integer tensor

    Returns
    -------
    z : tf.Tensor
        perturbed latent code on spherical space
    """
    # (batch_sz,)
    w = sample_w(kappa, d, batch_sz)
    w = tf.expand_dims(w, 1) * tf.ones((1, d), dtype=tf.float32)

    # (batch_sz, d)
    v = sample_v(mu, d, batch_sz)

    # (batch_sz, d)
    z = tf.multiply(
            v,
            tf.sqrt(tf.ones([batch_sz, d]) - tf.pow(w, 2)))\
        + tf.multiply(mu, w)


    return z
