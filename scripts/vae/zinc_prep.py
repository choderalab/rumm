import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()
from sklearn.preprocessing import StandardScaler
import pickle
import sys
sys.path.append('../rumm')
import lang
import nets
import bayesian

# constants
BATCH_SZ = 4096

zinc_df = pd.read_csv('res.csv', sep='\t')

# shuffle it, and conduct training-test split
zinc_df = zinc_df.sample(zinc_df.shape[0])
n_samples = zinc_df.shape[0]
n_tr = int(0.99 * n_samples)
y_tr = np.array(zinc_df.values[:n_tr, 2:-2], dtype=np.float32)
x_tr = zinc_df.values[:n_tr, -2]
y_te = np.array(zinc_df.values[n_tr:, 2:-2], dtype=np.float32)
x_te = zinc_df.values[n_tr:, -2]
x_tr = np.apply_along_axis(lambda x: 'G' + x + 'E', 0, x_tr)
x_te = np.apply_along_axis(lambda x: 'G' + x + 'E', 0, x_te)
fp_tr = np.array([list(line) for line in zinc_df.values[:n_tr, -1].tolist()], dtype='int')
fp_te = np.array([list(line) for line in zinc_df.values[n_tr:, -1].tolist()], dtype='int')
# calculate the std of y_tr for loss function
scaler = StandardScaler(copy=False)
y_tr = scaler.fit_transform(y_tr)
y_te = scaler.transform(y_te)
f_handle = open('scaler.p', 'wb')
pickle.dump(scaler, f_handle)
f_handle.close()

# save the dataset for later use
np.save('y_tr', y_tr)
np.save('y_te', y_te)
np.save('fp_tr', fp_tr)
np.save('fp_te', fp_te)

# create the language object and map it to strings
lang_obj = lang.Lang(list(x_tr + x_te))
vocab_size = len(lang_obj.idx2ch) + 1
x_tr = lang.preprocessing(x_tr, lang_obj)
x_te = lang.preprocessing(x_tr, lang_obj)
np.save('x_tr', x_tr)
np.save('x_te', x_te)
f_handle = open('lang_obj.p', 'wb')
pickle.dump(lang_obj, f_handle)
f_handle.close()
