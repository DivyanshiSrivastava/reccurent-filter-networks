"""
Apply recurrent neural filters to DNA sequence data.
This implementation uses Tensorflow and Keras.
"""

import numpy as np
import keras
import argparse
import os, sys
from subprocess import call

# sk-learn imports
from sklearn.metrics import average_precision_score as auprc
import sklearn
import sklearn.metrics
from sklearn.metrics import roc_auc_score

# keras imports
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D, LSTM, Reshape, GRU
from keras.layers import TimeDistributed
from keras.optimizers import SGD
from keras.engine import Layer
import keras.backend as K
from keras.models import load_model

# local imports
import utils as iu


# This class is adapted from:
# https://github.com/bloomberg/cnn-rnf/blob/master/cnn_keras.py
class DistributeInputLayer(Layer):
    """
        Distribute or break up the input DNA sequence of length L into chunks.
        Each chunk will be of size F, i.e. the recurrent neural filter size
        Each chunk i will be used as input to a filter at position i
        Range of i: (i=0 to i=(L-F+1))

        For example,
        if the filter length F=12, L=500,
        then i=0 to 489

        This inherits from the Keras Layer Class.

        Input dim: [batch_size x L x 4] # The 4 is for the one-hot-encoded dim.
        Output dim: [batch_size x (L - F + 1) x F x 4]
    """

    def __init__(self, filter_width, seq_len, **kwargs):
        super(DistributeInputLayer, self).__init__(**kwargs)
        self.filter_width = filter_width
        self.seq_len = seq_len

    def call(self, x):
        chunks = []
        print(self.seq_len - self.filter_width + 1)
        for start_idx in range(self.seq_len - self.filter_width + 1):
            chunk = x[:, start_idx: start_idx + self.filter_width]
            chunk = K.expand_dims(chunk, 1)
            chunks.append(chunk)
        input_chunks = keras.layers.concatenate(chunks, axis=1)
        dim_0_size = self.seq_len - self.filter_width + 1
        input_chunks = Reshape((dim_0_size, self.filter_width, 4))(input_chunks)
        print(input_chunks)
        print(input_chunks.shape)
        # Note: This shape should be (?, L-F+1, F, D)
        # L is the length of the input sequence, breaking it up into slices of
        # size F results in L-F+1 chunks.
        # For a DNA sequence of length 500, this is 489
        # F is the filter size/chunk size.
        # D is the depth (4)
        return input_chunks

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self.seq_len - self.filter_width + 1, self.filter_width, input_shape[-1])
        return output_shape


class RecurrentNeuralFilters:
    """
        Implementing the RNFs.
        The RNN or GRU filters are implemented using the TimeDistributed Layer.
        They are applied to each chunk obtained after applying the
        DistributeInput Layer.

        # Architecture:

        1. Distribute Inputs.
        2. Apply GRUs to each input chunk. (with time dist. layers.)

        Docs for time distributed layer is here:
        https://keras.io/api/layers/recurrent_layers/time_distributed/



    """
    def __init__(self, seq_length, rnf_filters, rnf_kernel_size, rnf_dim):
        self.seq_length = seq_length
        self.rnf_kernel_size = rnf_kernel_size
        self.rnf_filters = rnf_filters
        self.rnf_dim = rnf_dim

    def rnf_model(self):
        seq_input = Input(shape=(self.seq_length, 4,), name='seq')
        chunked_input = DistributeInputLayer(filter_width=self.rnf_kernel_size, seq_len=self.seq_length)(seq_input)
        # Shape:(?, L-F+1, F, D)
        # The TimeDistributed Layer treats index 1 in this input as \
        # independent time steps.
        # So here, the same GRU is being applied to every chunk.

        print('RNF_kernel_size: {}'.format(self.rnf_kernel_size))
        print('RNF_dimension: {}'.format(self.rnf_dim))

        xs = TimeDistributed(GRU(self.rnf_dim))(chunked_input)
        xs = Activation('relu')(xs)
        # Shape:(?, L-F+1, RNF_DIM) # Note here, the LSTM is producing
        # a single output with dimension RNF_DIM
        # Include an L-1 norm at the subsequent dense layer.
        xs = MaxPooling1D(pool_size=8, strides=4)(xs)
        print(xs.shape)
        # Adding Dense Layers.
        xs = Flatten()(xs)
        print(xs.shape)
        xs = Dense(128, activation='relu')(xs)
        print(xs.shape)
        xs = Dense(128, activation='relu')(xs)
        result = Dense(1, activation='sigmoid')(xs)
        # Define the model input & output
        model = Model(inputs=seq_input, outputs=result)
        return model


class MeasurePR(Callback):
    def on_train_begin(self, logs=None):
        self.val_auprc = []
        self.train_auprc = []

    def on_epoch_end(self, epoch, logs=None):
        """ monitor PR """
        x_val, y_val = self.validation_data[0], self.validation_data[1]
        predictions = self.model.predict(x_val)
        aupr = auprc(y_val, predictions)
        self.val_auprc.append(aupr)


class Train:
    """
    Defines all the methods used for training.
    """

    def __init__(self, training_data_path, val_data_path, test_data_path,
                 records_path, batchsize, seq_len, val_batchsize):

        # input/output paths
        self.training_data_path = training_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.records_path = records_path

        # other model parameters
        self.batchsize = batchsize
        self.seq_len = seq_len
        self.val_batchsize = val_batchsize

    def train_generator(self, path):
        X = iu.train_generator(path + ".seq",
                               self.batchsize, self.seq_len, 'seq', 'repeat')
        y = iu.train_generator(path + ".labels",
                               self.batchsize, self.seq_len, 'labels', 'repeat')
        while True:
            yield X.next(), y.next()

    def val_or_test_generator(self, path):
        X = iu.train_generator(path + ".seq",
                               self.val_batchsize, self.seq_len, 'seq',
                               'non-repeating')
        y = iu.train_generator(path + ".labels",
                               self.val_batchsize, self.seq_len, 'labels',
                               'non-repeating')
        while True:
            yield X.next(), y.next()

    def save_metrics(self, hist_object, pr_history):
        loss = hist_object.history['loss']
        val_loss = hist_object.history['val_loss']
        val_pr = pr_history.val_auprc

        # Saving the training metrics
        np.savetxt(self.records_path + 'trainingLoss.txt', loss, fmt='%1.2f')
        np.savetxt(self.records_path + 'valLoss.txt', val_loss, fmt='%1.2f')
        np.savetxt(self.records_path + 'valPRC.txt', val_pr, fmt='%1.2f')
        return loss, val_pr

    def return_best_model(self, pr_vec):
        # return the model with the lowest validation LOSS
        model_idx = np.argmax(pr_vec)
        # define the model path (The model files are 1-based)
        model_file = self.records_path + 'model_epoch' + str(model_idx + 1) + '.hdf5'
        # load and return the selected model:
        return load_model(model_file)

    def fit_model(self):
        # choose the parameters here:
        rnf_filters = [6, 12, 24]
        rnf_kernel_size = [6, 12, 16, 24]
        rnf_dim = [1, 2, 4]

        params = []
        # Performing a grid-search here
        for parameters in [rnf_filters, rnf_kernel_size, rnf_dim]:
            options = len(parameters)
            rnum = np.random.choice(options)
            params.append(parameters[rnum])

        architecture = RecurrentNeuralFilters(seq_length=self.seq_len,
                                              rnf_filters=params[0],
                                              rnf_kernel_size=params[1],
                                              rnf_dim=params[2])
        model = architecture.rnf_model()

        # Define the optimization here:
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd, metrics=['accuracy'])

        checkpointer = ModelCheckpoint(self.records_path + 'model_epoch{epoch}.hdf5',
                                       verbose=1, save_best_only=False)
        precision_recall_history = MeasurePR()
        # processing the generators
        train_generator = self.train_generator(self.training_data_path)
        val_data = self.val_or_test_generator(self.val_data_path).next()

        # calculating steps.
        training_set_size = len(np.loadtxt(self.training_data_path + '.labels'))
        # Calculate the steps per epoch
        steps = training_set_size / self.batchsize

        hist = model.fit_generator(epochs=10, steps_per_epoch=steps,
                                   generator=train_generator,
                                   validation_data=val_data,
                                   callbacks=[checkpointer, precision_recall_history])

        loss, val_pr = self.save_metrics(hist, precision_recall_history)
        model = self.return_best_model(pr_vec=val_pr)
        return loss, val_pr, model


class Evaluate:
    """
    pass
    """

    def __init__(self, model, batchsize, seq_len, test_data_path, records_path):
        self.model = model
        self.batchsize = batchsize
        self.seq_len = seq_len
        self.test_data_path = test_data_path
        self.records_path = records_path

    def test_or_val_generator(self, path):
        X = iu.train_generator(path + ".seq",
                               self.batchsize, self.seq_len, 'seq',
                               'non-repeating')
        y = iu.train_generator(path + ".labels",
                               self.batchsize, self.seq_len, 'labels',
                               'non-repeating')
        while True:
            yield X.next(), y.next()

    def evaluate(self):
        test_generator = self.test_or_val_generator(self.test_data_path)
        probabilities = self.model.predict_generator(test_generator, 5000)
        test_labels = np.loadtxt(self.test_data_path + '.labels')

        # Calculate auROC

        roc_auc = roc_auc_score(test_labels, probabilities)
        # Calculate auPRC
        prc_auc = auprc(test_labels, probabilities)
        self.records_path.write('')
        # Write auROC and auPRC to records file.
        self.records_path.write("AUC ROC:{0}\n".format(roc_auc))
        self.records_path.write("AUC PRC:{0}\n".format(prc_auc))


def main():

    parser = argparse.ArgumentParser(description='RNF-simplified')
    parser.add_argument('training_data_path')
    parser.add_argument('validation_data_path')
    parser.add_argument('test_data_path')
    parser.add_argument('seq_len')
    parser.add_argument('batchsize')
    parser.add_argument('val_size')
    parser.add_argument('outfile', help='outfile with model performance metrics')

    args = parser.parse_args()
    # Create output directory:
    outdir = args.outfile
    call(['mkdir', outdir])

    tr = Train(args.training_data_path, args.validation_data_path,
               args.test_data_path, args.outfile, int(args.batchsize),
               int(args.seq_len), int(args.val_size))
    _, _, model = tr.fit_model()
    te = Evaluate(model, args.batchsize, int(args.seq_len),
                  args.test_data_path, args.outfile)
    te.evaluate()


if __name__ == '__main__':
    main()