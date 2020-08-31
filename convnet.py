"""
This code implements a standard CNN; used for testing whether the randomized
training and validation schema work.
If this performs at par with BichomSEQ on the ENCODE sets; then move onto
using this same schema with recurrent nueral filters.
Note: The code structure should remain similar for the RNF architectures.
"""

import subprocess

# sk-learn imports
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# keras imports
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Reshape
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import Callback


import get_data


# using a callback to access validation data and access auPRC at each
# epoch
class PrecisionRecall(Callback):
    def on_train_begin(self, logs=None):
        self.val_auprc = []
        self.train_auprc = []

    def on_epoch_end(self, epoch, logs=None):
        """ monitor PR """
        x_val, y_val = self.validation_data[0], self.validation_data[1]
        predictions = self.model.predict(x_val)
        au_prc = average_precision_score(y_val, predictions)
        self.val_auprc.append(au_prc)


class ConvNet:

    def __init__(self, window_len, n_filters, filter_size, pooling_stride,
                 pooling_size, n_dense_layers, dropout_freq, dense_size):
        self.window_len = window_len
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pooling_stride = pooling_stride
        self.pooling_size = pooling_size
        self.n_dense_layers = n_dense_layers
        self.dropout_freq = dropout_freq
        self.dense_size = dense_size

    def get_model(self):
        """
        Define the architecture; standard 1-D CNN (Bichom-SEQ)
        """
        seq_input = Input(shape=(self.window_len, 4,), name='seq')
        xs = Conv1D(filters=self.n_filters, kernel_size=self.filter_size,
                    padding='same')(seq_input)
        xs = Activation('relu')(xs)
        xs = Conv1D(filters=self.n_filters, kernel_size=self.filter_size,
                    padding='same')(xs)
        xs = Activation('relu')(xs)
        xs = MaxPooling1D(padding="same", strides=self.pooling_stride,
                          pool_size=self.pooling_size)(xs)
        xs = LSTM(32, activation='relu')(xs)
        for idx in range(self.n_dense_layers):
            # adding in Dense Layers
            xs = Dense(self.dense_size, activation='relu')(xs)
            xs = Dropout(self.dropout_freq)(xs)
        result = Dense(1, activation='sigmoid')(xs)
        model = Model(inputs=seq_input, outputs=result)
        return model

    def fit_the_data(self, model_cnn, train_gen, val_data, patience,
                     steps_per_epoch):
        # fit the data
        adam = Adam(learning_rate=0.001)
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model_cnn.compile(loss='binary_crossentropy',
                          optimizer=adam, metrics=['accuracy'])
        earlystop = EarlyStopping(monitor='val_loss', mode='min',
                                  verbose=1, min_delta=0.01, patience=patience)
        model_cnn.fit(train_gen,
                      steps_per_epoch=steps_per_epoch,
                      epochs=100,
                      validation_data=val_data,
                      callbacks=[earlystop])

        return model_cnn

    def evaluate_and_save_model(self, model, test_data_tuple, results_dir):
        # note this tuple is built in process_data.py
        x_test, y_test, bed_coords_test = test_data_tuple
        model_probas = model.predict(x_test)
        auroc = roc_auc_score(y_test, model_probas)
        auprc = average_precision_score(y_test, model_probas)

        subprocess.call(['mkdir', results_dir])
        records_file = results_dir + '/metrics.txt'

        with open(records_file, "w") as rf:
            # save metrics to results file in the outdir:
            rf.write("Model:{0}\n".format('cnn'))
            rf.write("AUC ROC:{0}\n".format(auroc))
            rf.write("AUC PRC:{0}\n".format(auprc))

        model.save(results_dir + '/model.hdf5')
        return auroc, auprc


def train_model(genome_size, fa, peaks, blacklist, results_dir, batch_size,
                steps, patience, acc_regions_file):

    print(steps)

    print('getting the generators & test dataset')
    train_generator, val_data, test_data = \
            get_data.get_train_and_val_generators(genome_sizes=genome_size,
                                                  fa=fa,
                                                  peaks=peaks,
                                                  blacklist=blacklist,
                                                  batch_size=batch_size,
                                                  acc_regions_file=acc_regions_file)

    print('building convolutional architecture')
    architecture = ConvNet(window_len=500, n_filters=128, filter_size=24,
                           pooling_stride=8, pooling_size=8, n_dense_layers=3,
                           dropout_freq=0.5, dense_size=128)
    model = architecture.get_model()
    print('fitting the model')
    # parsing the validation data tuple. (same as that returned be test data)
    x_val, y_val, bed_coords_val = val_data
    fitted_model = architecture.fit_the_data(model_cnn=model,
                                             train_gen=train_generator,
                                             val_data=(x_val, y_val),
                                             patience=patience,
                                             steps_per_epoch=steps)
    print('evaluating the model')
    architecture.evaluate_and_save_model(model=fitted_model,
                                         test_data_tuple=test_data,
                                         results_dir=results_dir)






