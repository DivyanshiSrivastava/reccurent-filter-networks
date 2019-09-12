import numpy as np
import keras

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# keras imports
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D, LSTM, Reshape
from keras.layers import TimeDistributed
from keras.optimizers import SGD
from keras.layers import Lambda

# local imports
from simulatedata import TrainingData, TestData


def make_onehot(buf, seq_length):

    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
          'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1],
          'N': [0, 0, 0, 0]}

    one_hot = [fd[base] for seq in buf for base in seq]
    one_hot_np = np.reshape(one_hot, (-1, seq_length, 4))
    return one_hot_np


class ConvModel:

    def __init__(self, seq_length, conv_layers, conv_filters, conv_kernel_size,
                 dense_nodes):
        self.seq_length = seq_length
        self.conv_layers = conv_layers
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.dense_nodes = dense_nodes

    def convolution_model(self):
        seq_input = Input(shape=(self.seq_length, 4,), name='seq')

        assert self.conv_layers > 0

        for idx in range(self.conv_layers):
            # Adding in convolution layers
            layer_name = 'convolution' + str(idx + 1)
            xs = Conv1D(filters=self.conv_filters, kernel_size=self.conv_kernel_size,
                        padding='same', name=layer_name)(seq_input)
            xs = Activation('relu')(xs)

        xs = MaxPooling1D(padding='same', strides=15, pool_size=15)(xs)
        xs = Flatten()(xs)
        # 2 FC dense layers
        xs = Dense(self.dense_nodes, activation='relu')(xs)
        xs = Dropout(0.5)(xs)
        xs = Dense(self.dense_nodes, activation='relu')(xs)
        xs = Dropout(0.5)(xs)
        # Output
        result = Dense(1, activation='sigmoid')(xs)
        # Define the model input & output
        model = Model(inputs=seq_input, outputs=result)
        return model


class RNF:

    def __init__(self, seq_length, rnf_filters, rnf_kernel_size, conv_filters, conv_kernel_size,
                 dense_nodes):
        self.seq_length = seq_length
        self.rnf_kernel_size = rnf_kernel_size
        self.rnf_filters = rnf_filters
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.dense_nodes = dense_nodes

    @staticmethod
    def slice_input(start, end):
        # Crops (or slices) a Tensor along dimension 1 from start to end
        def func(x):
            return x[:, start: end]
        return Lambda(func)

    def rnf_model(self):
        seq_input = Input(shape=(self.seq_length, 4,), name='seq')

        # Defining RNF filters
        def rnn_filters(idx):
            # Start scanning here
            start_idx = 0
            size = self.rnf_kernel_size
            # Define convolutional stride
            step = 1

            shared_layer = LSTM(1, name='LSTM' + str(idx))
            # Store outputs here:
            input_chunks = []
            # Defining a 'valid' convolution here:
            while start_idx + size < self.seq_length:
                sliced_input = self.slice_input(start_idx, start_idx + size)(seq_input)
                print sliced_input.shape
                input_chunks.append(sliced_input)
                start_idx += step

            input_chunks = keras.layers.concatenate(input_chunks, axis=1)
            print input_chunks.shape

            dim_0_size = self.seq_length - self.rnf_kernel_size
            input_chunks = Reshape((dim_0_size, self.rnf_kernel_size, 4))(input_chunks)
            print input_chunks.shape
            xs = TimeDistributed(shared_layer)(input_chunks)
            print xs.shape
            return xs

        filter_outs = []
        # Using the parameter for the number of filters.
        for idx in range(self.rnf_filters):
            filter_outs.append(rnn_filters(idx))

        xs = keras.layers.concatenate(filter_outs)
        xs = Activation('relu')(xs)
        xs = Conv1D(filters=self.conv_filters, kernel_size=self.conv_kernel_size)(xs)
        xs = Activation('relu')(xs)
        # From here on, the model is identical to the convolutional model.
        xs = MaxPooling1D(padding='same', strides=15, pool_size=15)(xs)
        xs = Flatten()(xs)
        # 2 FC dense layers
        xs = Dense(self.dense_nodes, activation='relu')(xs)
        xs = Dropout(0.5)(xs)
        xs = Dense(self.dense_nodes, activation='relu')(xs)
        xs = Dropout(0.5)(xs)
        # Output
        result = Dense(1, activation='sigmoid')(xs)
        # Define the model input & output
        model = Model(inputs=seq_input, outputs=result)
        return model


def fit_model(model_type, dat, labels, batch_size):

    if model_type == 'conv':
        architecture = ConvModel(seq_length=100, conv_layers=10, conv_filters=64,
                                 conv_kernel_size=24, dense_nodes=32)
        model = architecture.convolution_model()
    else:
        architecture = RNF(seq_length=100, rnf_filters=6, rnf_kernel_size=12,
                           conv_filters=64, conv_kernel_size=24, dense_nodes=32)
        model = architecture.rnf_model()

    X_train, X_test, y_train, y_test = train_test_split(dat, labels.astype(int))

    # Define the optimization here:
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', mode='min',
                              verbose=1, patience=16)
    model.fit(x=X_train, y=y_train, epochs=32, batch_size=batch_size,
              validation_split=0.2, callbacks=[earlystop])

    probas = model.predict(X_test)
    auroc = roc_auc_score(y_test, probas)
    auprc = average_precision_score(y_test, probas)
    print auroc, auprc
    return model


def main():

    # Defining the motifs
    motif_a = 'TAATGA'
    motif_b = 'TAATCT'
    motif_flipped = 'TAATCA'
    motif_random = 'AAAAAA'

    # Instantiate an instance of TrainingData
    # Change N / Test over many N's.
    train_data = TrainingData(motif_a=motif_a, motif_b=motif_b, N=10000, seq_length=100)

    # Getting the synthetic X and y data
    dat, labels = train_data.simulate_data()
    model = fit_model(model_type='conv', dat=dat, labels=labels, batch_size=64)  # Keeping the batch size fixed for now.

    # Instantiating an instance of Evaluating Composition

    td = TestData(seq_length=100, model=model)
    # Testing across 4 motifs
    print td.simulate_test_dat(motif_a)
    print td.simulate_test_dat(motif_b)
    print td.simulate_test_dat(motif_flipped)
    print td.simulate_test_dat(motif_random)


if __name__ == '__main__':
    main()

