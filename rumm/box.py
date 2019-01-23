"""
box.py

Implements a black box wrapper to wrap up the models, the flow.
Also used to save and restore weights.
"""

# imports
import tensorflow as tf
import numpy as np
import logging
from typing import *

class Box:
    """The wrapper of all layers in the model."""

    def __init__(self, flow, models: list,
                 n_epochs=10, batch_sz=1,
                 loss_fn=tf.losses.mean_squared_error):
        self.flow = flow
        self.models = models
        self.n_epochs = n_epochs
        self.batch_sz = batch_sz
        self.loss_fn = loss_fn
        for model in self.models:
            if hasattr(model, 'batch_sz'):
                model.batch_sz = self.batch_sz

    def train(self, x_tr, y_tr,
                optimizer=tf.train.AdamOptimizer(),
                loss_fn=tf.losses.mean_squared_error):
        """Train the model with training set, x and y

        Parameters
        ----------
        x_tr : tf.Tensor or np.ndarray, training data
        y_tr : tf.Tensor or np.ndarray, training labels
        optimizer :
            Default value = tf.train.AdamOptimizer())
        loss_fn :
            Default value = tf.losses.mean_squared_error)

        Returns
        -------

        """
        # convert them into tensors.
        x_tr = tf.convert_to_tensor(x_tr)
        y_tr = np.array(y_tr, dtype=np.float32)
        if y_tr.ndim == 1 or y_tr.shape[1] == 1:
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
                    logging.info("epoch %s batch %s loss %s" % (epoch, batch, np.asscalar(loss.numpy())))

    def predict(self, x_te):
        """Make predictions on the x of test set.

        Parameters
        ----------
        x_te : tf.Tensor or np.ndarray, test data


        Returns
        -------

        """

        # this is necessary in order to go through all the samples in test set
        for model in self.models:
            if hasattr(model, 'batch_sz'):
                model.batch_sz = 1

        ys_hat_all = np.array([])
        x_te = tf.convert_to_tensor(x_te)
        ds_te = tf.data.Dataset.from_tensor_slices((x_te))
        # ds_te = ds_te.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_sz))
        ds_te = ds_te.apply(tf.contrib.data.batch_and_drop_remainder(1))
        for xs in ds_te:
            ys_hat = self.flow(xs, self.models)
            ys_hat_all = np.concatenate([ys_hat_all, ys_hat.numpy().flatten()], axis=0)
        return ys_hat_all

    def save_weights(self, file_path):
        """Save the model. Note that it is necessary to also save the shape of the input.

        Parameters
        ----------
        file_path : str, 


        Returns
        -------

        """
        import os
        os.system('rm -rf ' + file_path)
        os.system('mkdir ' + file_path)
        for idx, model in enumerate(self.models):
            model.save_weights('%s/%s.h5' % (file_path, idx))
            # model.save_weights('%s/%s.h5' % (file_path, idx))

    def load_weights(self, file_path):
        """Restore the model.

        Parameters
        ----------
        file_path :


        Returns
        -------

        """
        for idx, model in self.models:
            model.load_weights('%s/%s.h5' % (file_path, idx))

class HyperParamTuningBox():
    """Handles the tuning of hyperparameters."""
    def __init__(self, flow, models, param_grid, model_update_fn,
                 n_epochs=10, batch_sz=1,
                 loss_fn=tf.losses.mean_squared_error):
        self.flow = flow
        self.models = models
        self.param_grid = param_grid
        self.model_update_fn = model_update_fn
        self.n_epochs = n_epochs
        self.batch_sz = batch_sz
        self.loss_fn = loss_fn
        for model in self.models:
            if hasattr(model, 'batch_sz'):
                model.batch_sz = self.batch_sz

    def load_dataset(self, df, smiles_idx: int, prop_idx: int) -> None:
        """Load a pandas DataFrame.

        Parameters
        ----------
        df :
            param smiles_idx: int:
        prop_idx :
            int:
        smiles_idx: int :

        prop_idx: int :


        Returns
        -------

        """
        # take the rows of interest
        x_all = df.values[smiles_idx]
        y_all = df.values[prop_idx]

        self.x_all = x_all
        self.y_all = y_all

    def split(self, split_param: List[float], x, y, random=True):
        """Split the set into training and test.

        Parameters
        ----------
        split_param :
            type split_param: list of floats, could be in the length of two or three.
        x :
            type x: np.array of SMILES strings to split from
        y :
            type y: np.array of properties to split from
        split_param :
            List[float]:
        random :
            Default value = True)
        split_param: List[float] :


        Returns
        -------

        """

        if random == True: # shuffle the data
            import random
            idxs = range(y.shape[0])
            random.shuffle(idxs)
            x = np.take(x, idxs)
            y = np.take(y, idxs)

        assert(y.shape[0] == x.shape[0], "x and y must have the same first dimension")
        n_samples = y.shape[0]
        # only split into training and test
        if len(split_param) == 2:
            n_tr = int(n_samples * split_param[0])
            x_tr = x[:n_tr]
            y_tr = y[:n_tr]
            x_te = x[n_tr:]
            y_te = y[n_tr:]
            return x_tr, y_tr, x_te, y_te

        elif len(split_param) == 3:
            n_tr = int(n_samples * split_param[0])
            n_te = int(n_samples * split_param[1])
            x_tr = x[:n_tr]
            y_tr = y[:n_tr]
            x_te = x[n_tr:(n_tr + n_te)]
            y_te = y[n_tr:(n_tr + n_te)]
            x_val = x[(n_tr + n_te):]
            y_val = y[(n_tr + n_te):]
            return x_tr, y_tr, x_te, y_te, x_val, y_val

        else:
            raise NotImplementedError


    def grid_search(self):
        """The function that handles the tuning of the cross training and tuning of
        the hyperparameters.

        Parameters
        ----------

        Returns
        -------

        """
        x_tr, y_tr, x_te, y_te, x_val, y_val = self.split([0.8, 0.1, 0.1],
                                               self.x_all, self.y_all)

        raise NotImplementedError
