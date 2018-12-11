"""
Test bayesian facility

"""

# imports
import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()

import sys
sys.path.append('..')
import bayesian

def test_mdl():
    mdl = bayesian.MixDistributionLayer(3, 16)
    x = [np.random.rand(16)]
    x = tf.convert_to_tensor(x)
    ds = mdl(x)
    print(ds.sample(1000))
    print(ds.log_prob(np.random.rand(16)))

test_mdl()
