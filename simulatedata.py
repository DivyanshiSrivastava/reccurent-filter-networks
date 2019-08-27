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
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD


def make_onehot(buf, seq_length):
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    onehot = [fd[base] for seq in buf for base in seq]
    onehot_np = np.reshape(onehot, (-1, seq_length, 4))
    return onehot_np


def embed(sequence):
    pos = np.random.randint(5, 85)  # choose the insert position
    sequence[pos: pos+8] = 'TGATTTAT'


def simulate_data():
    # define a integer --> str dictionary
    letter = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
    # Simulating the negative data
    # These are 1000 sequences with background frequencies A/T=0.5
    seq_list = []
    for idx in range(1000):
        sequence = np.random.randint(0, 4, 100)
        sequence = ''.join([letter[x] for x in sequence])
        seq_list.append((sequence, 0))  # The 0 here is the sequence label
    # Simulating the positive data
    # These are 1000 sequences with an embedded motif at any position
    for idx in range(1000):
        sequence = np.random.randint(0, 4, 100)
        sequence = [letter[x] for x in sequence]
        embed(sequence)
        seq_list.append((''.join(sequence), 1))  # Doing the join after the embedding for the positive set
    # extracting the sequence data
    dat = np.array(seq_list)[:, 0]
    dat = make_onehot(dat, seq_length=100)
    labels = np.array(seq_list)[:, 1]
    return dat, labels


def build_model():
    """ Define a Keras graph model with sequence and accesibility as input """
    seq_input = Input(shape=(100, 4,), name='seq')
    xs = Conv1D(16, 64, padding="same")(seq_input)
    xs = Activation('relu')(xs)
    xs = MaxPooling1D(padding="same", strides=15, pool_size=15)(xs)
    xs = Flatten()(xs)
    # fully connected dense layers
    xs = Dense(32, activation='relu')(xs)
    xs = Dropout(0.75)(xs)
    xs = Dense(16, activation='relu')(xs)
    result = Dense(1, activation='sigmoid')(xs)
    # define the model input & output
    model = Model(inputs=seq_input, outputs=result)
    return model


def fit_model(dat, labels):
    model = build_model()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x=dat, y=labels, epochs=50, batch_size=32)
    # save the model
    model.save('/Users/divyanshisrivastava/Desktop/model.hdf5')
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
    model = load_model('/Users/divyanshisrivastava/Desktop/model.hdf5')
    evaluate_model(model, X_test, y_test.astype(int))


if __name__ == '__main__':
    main()

