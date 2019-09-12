import numpy as np
import sys, os
import keras

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# keras imports
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D, LSTM, Reshape, SimpleRNN
from keras.layers import TimeDistributed
from keras.optimizers import SGD
from keras.layers import Lambda
import keras.backend as K


def make_onehot(buf, seq_length):
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    onehot = [fd[base] for seq in buf for base in seq]
    onehot_np = np.reshape(onehot, (-1, seq_length, 4))
    return onehot_np


def embed(sequence):
    pos = np.random.randint(5, 85)  # choose the insert position

    # Draw from a uniform 0, 100
    choice = np.random.randint(0, 100)
    # Random chance here, testing for transitions
    if choice < 50:
        sequence[pos: pos+8] = 'CAGCTGAA'
    else:
        sequence[pos: pos+8] = 'CAGCTGGG'


def simulate_data():
    # define a integer --> str dictionary
    letter = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
    # Simulating the negative data
    # These are 1000 sequences with background frequencies A/T=0.5
    seq_list = []
    for idx in range(5000):
        sequence = np.random.randint(0, 4, 100)
        sequence = ''.join([letter[x] for x in sequence])
        seq_list.append((sequence, 0))  # The 0 here is the sequence label
    # Simulating the positive data
    # These are 1000 sequences with an embedded motif at any position
    for idx in range(5000):
        sequence = np.random.randint(0, 4, 100)
        sequence = [letter[x] for x in sequence]
        embed(sequence)
        seq_list.append((''.join(sequence), 1))  # Doing the join after the embedding for the positive set
    # extracting the sequence data
    dat = np.array(seq_list)[:, 0]
    dat = make_onehot(dat, seq_length=100)
    labels = np.array(seq_list)[:, 1]
    return dat, labels


def embed_test_motif(sequence, motif):
    pos = np.random.randint(5, 85)  # choose the insert position
    sequence[pos: pos + 8] = motif


def simulate_test_dat(motif):    # define a integer --> str dictionary
    letter = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
    # Simulating the negative data
    # These are 1000 sequences with background frequencies A/T=0.5
    seq_list = []

    for idx in range(1000):
        sequence = np.random.randint(0, 4, 100)
        sequence = [letter[x] for x in sequence]
        embed_test_motif(sequence, motif)
        seq_list.append((''.join(sequence), 1))  # Doing the join after the embedding for the positive set
    # extracting the sequence data
    dat = np.array(seq_list)[:, 0]
    dat = make_onehot(dat, seq_length=100)
    labels = np.array(seq_list)[:, 1]
    return dat, labels


# MODEL HERE
# Code adapted from:
# https://github.com/keras-team/keras/issues/890

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


def build_model():
    """ Define a Keras graph model with sequence and accesibility as input """
    seq_input = Input(shape=(100, 4,), name='seq')
    print seq_input.shape

    def rnn_filt(idx):
        lstm_outs = []
        start_idx = 0
        size = 10
        step = 1
        seq_length = 90
        shared_layer = LSTM(1, name='RNN' + str(idx))
        input_chunks = []
        while start_idx + step <= seq_length:
            sliced_input = crop(1, start_idx, start_idx + size)(seq_input)
            input_chunks.append(sliced_input)
            start_idx += step
        input_chunks = keras.layers.concatenate(input_chunks, axis=1)
        print input_chunks.shape
        input_chunks = Reshape((90, 10, 4))(input_chunks)
        print input_chunks.shape
        xs = TimeDistributed(shared_layer)(input_chunks)
        print xs.shape
        return xs

    filter_outs = []
    for idx in range(12):
        filter_outs.append(rnn_filt(idx))

    xs = keras.layers.concatenate(filter_outs)
    xs = Activation('relu')(xs)

    xs = Conv1D(filters=32, kernel_size=10)(xs) # These parameters are controlling this?
    # xs = Conv1D(filters=64, kernel_size=16)(xs) # Are these parameters also controll
    xs = Activation('relu')(xs)
    xs = MaxPooling1D(pool_size=10)(xs)
    print xs.shape
    xs = Flatten()(xs)
    # fully connected dense layers
    xs = Dense(16, activation='relu')(xs)
    result = Dense(1, activation='sigmoid')(xs)
    # define the model input & output
    model = Model(inputs=seq_input, outputs=result)
    return model


def fit_model(dat, labels):
    model = build_model()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x=dat, y=labels, epochs=10, batch_size=64)
    # save the model
    model.save('/Users/asheesh/Desktop/model.hdf5')
    return model


def evaluate_model(model, test_dat, test_labels):
    probas = model.predict(test_dat)
    auroc = roc_auc_score(test_labels, probas)
    auprc = average_precision_score(test_labels, probas)
    print "auroc:", auroc
    print "auc(pr):", auprc


def main():
    dat, labels = simulate_data()
    X_train, X_test, y_train, y_test = train_test_split(dat, labels)
    model = fit_model(X_train, y_train.astype(int))
    model = load_model('/Users/asheesh/Desktop/model.hdf5')
    evaluate_model(model, X_test, y_test.astype(int))

    test_dat_m, test_lab_m = simulate_test_dat('CAGCTGAA')
    print np.mean(model.predict(test_dat_m))

    test_dat_m, test_lab_m = simulate_test_dat('CAGCTGGG')
    print np.mean(model.predict(test_dat_m))

    test_dat_m, test_lab_m = simulate_test_dat('CAGCTGAG')
    print np.mean(model.predict(test_dat_m))

    test_dat_m, test_lab_m = simulate_test_dat('CCCCCCCC')
    print np.mean(model.predict(test_dat_m))


if __name__ == '__main__':
    main()

