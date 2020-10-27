import subprocess
import pybedtools

# sk-learn imports
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# keras imports
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Reshape
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer, TimeDistributed, SimpleRNN
import tensorflow.keras.backend as K

import get_data
import convnet


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
        input_chunks = tf.keras.layers.concatenate(chunks, axis=1)
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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'seq_len': self.seq_len,
            'filter_width': self.filter_width
        })
        return config


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

    def __init__(self, window_len, pooling_stride, pooling_size, n_dense_layers,
                 dropout_freq, dense_size, rnf_kernel_size, n_filters):
        self.window_len = window_len  # previously seq_length
        self.pooling_stride = pooling_stride
        self.pooling_size = pooling_size
        self.n_dense_layers = n_dense_layers
        self.dropout_freq = dropout_freq
        self.dense_size = dense_size
        # RF parameters:
        self.rnf_kernel_size = rnf_kernel_size
        self.n_filters = n_filters  # previously: rnf_filters
        # Note: rnf_kernel_size and n_filters are same as that for the convnet.

    def build_rnf_model(self):
        seq_input = Input(shape=(self.window_len, 4,), name='seq_input')
        chunked_input = DistributeInputLayer(filter_width=self.rnf_kernel_size,
                                             seq_len=self.window_len)(seq_input)
        # Shape:(?, L-F+1, F, D)
        # The TimeDistributed Layer treats index 1 in this input as \
        # independent time steps.
        # So here, the same GRU is being applied to every chunk.
        print('RNF_kernel_size: {}'.format(self.rnf_kernel_size))
        print('RNF_dimension: {}'.format(self.n_filters))
        xs = TimeDistributed(SimpleRNN(self.n_filters))(chunked_input)
        xs = Activation('relu')(xs)
        # Shape:(?, L-F+1, RNF_DIM) # Note here, the LSTM is producing
        # a single output with dimension RNF_DIM
        # Include an L-1 norm at the subsequent dense layer.
        xs = MaxPooling1D(pool_size=15, strides=15)(xs)
        print(xs.shape)
        # Adding Dense Layers.
        xs = LSTM(32, activation='relu')(xs)
        print(xs.shape)
        xs = Dense(128, activation='relu')(xs)
        xs = Dropout(self.dropout_freq)(xs)
        print(xs.shape)
        xs = Dense(128, activation='relu')(xs)
        xs = Dropout(self.dropout_freq)(xs)
        result = Dense(1, activation='sigmoid')(xs)
        # Define the model input & output
        model = Model(inputs=seq_input, outputs=result)
        return model

    def fit_the_data(self, model_rnf, train_gen, val_data, patience,
                     steps_per_epoch, opt, learning_rate, results_dir):
        # fit the data
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=steps_per_epoch * 2,
            decay_rate=0.96, staircase=False, name=None)        #
        adam = Adam(learning_rate=lr_schedule)
        sgd = SGD(lr=learning_rate, decay=0.001, momentum=0.9, nesterov=True)
        if opt == 'adam':
            optimizer = adam
        else:
            optimizer = sgd

        auprc = tf.keras.metrics.AUC(curve='PR')
        model_rnf.compile(loss='binary_crossentropy',
                          optimizer=optimizer, metrics=['accuracy', auprc])
        precision_recall_history = convnet.PrecisionRecall(val_data=val_data)
        earlystop = EarlyStopping(monitor='val_loss', mode='min',
                                  verbose=1, min_delta=0.001, patience=patience)
        # save the best performing model
        model_path_best = results_dir + '/model.best.hdf5'
        checkpoint_models = tf.keras.callbacks.ModelCheckpoint(filepath=model_path_best,
                                                               monitor='val_auc',
                                                               save_best_only=True)
        model_rnf.fit(train_gen,
                      steps_per_epoch=steps_per_epoch,
                      epochs=25,
                      validation_data=val_data,
                      callbacks=[earlystop, precision_recall_history, checkpoint_models])
        print(precision_recall_history.val_auprc)
        # also return the best model
        best_model_pr = load_model(model_path_best)
        return model_rnf, best_model_pr

    def evaluate_and_save_model(self, model, test_data_tuple, results_dir,
                                outname):
        # note this tuple is built in process_data.py
        x_test, y_test, bed_coords_test = test_data_tuple
        model_probas = model.predict(x_test)
        auroc = roc_auc_score(y_test, model_probas)
        auprc = average_precision_score(y_test, model_probas)

        subprocess.call(['mkdir', results_dir])
        records_file = results_dir + '/' + outname + '.metrics.txt'

        with open(records_file, "w") as rf:
            # save metrics to results file in the outdir:
            rf.write("Model:{0}\n".format('cnn'))
            rf.write("AUC ROC:{0}\n".format(auroc))
            rf.write("AUC PRC:{0}\n".format(auprc))

        model.save(results_dir + '/' + outname + '.hdf5')
        return auroc, auprc


def train_model(genome_size, fa, peaks, blacklist, results_dir, batch_size,
                steps, patience, acc_regions_file, learning_rate, opt,
                ratios, filter_width, no_of_filters):
    print(steps)
    subprocess.call(['mkdir', results_dir])
    print('getting the generators & test dataset')
    train_generator, val_data, test_data = \
        get_data.get_train_and_val_generators(genome_sizes=genome_size,
                                              fa=fa,
                                              peaks=peaks,
                                              blacklist=blacklist,
                                              batch_size=batch_size,
                                              acc_regions_file=acc_regions_file,
                                              ratios=ratios)

    print('building convolutional architecture')
    architecture = RecurrentNeuralFilters(window_len=500,
                                          n_filters=no_of_filters,
                                          rnf_kernel_size=filter_width,
                                          pooling_stride=15,
                                          pooling_size=15, n_dense_layers=2,
                                          dropout_freq=0.5, dense_size=128)
    model = architecture.build_rnf_model()
    print('fitting the model')
    # parsing the validation data tuple. (same as that returned be test data)
    x_val, y_val, bed_coords_val = val_data
    fitted_model, best_model = architecture.fit_the_data(model_rnf=model,
                                                         train_gen=train_generator,
                                                         val_data=(
                                                         x_val, y_val),
                                                         patience=patience,
                                                         steps_per_epoch=steps,
                                                         learning_rate=learning_rate,
                                                         opt=opt,
                                                         results_dir=results_dir)

    print('evaluating the model')
    architecture.evaluate_and_save_model(model=fitted_model,
                                         test_data_tuple=test_data,
                                         results_dir=results_dir,
                                         outname='model')
    architecture.evaluate_and_save_model(model=best_model,
                                         test_data_tuple=test_data,
                                         results_dir=results_dir,
                                         outname='model_top')