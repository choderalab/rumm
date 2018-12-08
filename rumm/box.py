"""
box.py

Implements a black box wrapper to wrap up the models, the flow.
Also used to save and restore weights.
"""

# imports
import tensorflow as tf
import numpy as np


class Box:
    """
    The wrapper of all layers in the model.

    Parameters
    ----------
    models : a nested structure of models
    n_epochs : int, number of epochs
    batch_sz : int, batch size
    loss_fn : a function that describe the loss during training
    """

    def __init__(self, flow, models, n_epochs=10, batch_sz=1, loss_fn=tf.losses.mean_squared_error):
        self.flow = flow
        self.models = models
        self.n_epochs = n_epochs
        self.batch_sz = batch_sz
        self.loss_fn = loss_fn
        for model in models:
            if hasattr(model, 'batch_sz'):
                model.batch_sz = self.batch_sz

    def train(self, x_tr, y_tr,
                optimizer=tf.train.AdamOptimizer(),
                loss_fn=tf.losses.mean_squared_error):
        """
        Train the model with training set, x and y

        Parameters
        ----------
        flow : the function which takes the data and the model, to make a prediction
        x_tr : np.ndarry
        y_tr : np.ndarry, has to match the number of samples in x_tr
        """
        # convert them into tensors
        y_tr = np.array(y_tr, dtype=np.float32)
        x_tr = tf.convert_to_tensor(x_tr)

        if y_tr.ndims == 1 or y_tr.shape[1] == 1:
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
                    print("epoch %s batch %s loss %s" % (epoch, batch, np.asscalar(loss.numpy())))

    def predict(self, x_te):
        """
        Make predictions on the x of test set.

        Parameters
        ----------
        flow : the function that takes data and the models and output ys
        x_te : np.ndarry, the test data
        """

        ys_hat_all = np.array([])
        x_te = tf.convert_to_tensor(x_te)
        ds_te = tf.data.Dataset.from_tensor_slices((x_te))
        ds_te = ds.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_sz))
        for xs in ds_te:
            ys_hat = self.flow(xs, self.models)
            ys_hat_all = np.concatenate([ys_hat_all, ys_hat.numpy().flatten()], axis=0)
        return ys_hat_all

    def save_weights(self, file_path):
        """
        Save the model. Note that it is necessary to also save the shape of the input.
        """
        import os
        os.system('rm -rf ' + file_path)
        os.system('mkdir ' + file_path)
        for idx, model in enumerate(self.models):
            model.save_weights('%s/%s.h5' % (file_path, idx))
            # model.save_weights('%s/%s.h5' % (file_path, idx))

    def load_weights(self, file_path):
        """
        Restore the model.
        """
        for idx, model in self.models:
            model.load_weights('%s/%s.h5' % (file_path, idx))
