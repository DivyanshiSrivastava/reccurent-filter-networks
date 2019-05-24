import numpy as np
import sys, os
import keras

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# keras imports
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, Input, LSTM, Bidirectional, SimpleRNN
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD

# local imports
from integratedgrads import get_sequence_attribution

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns


def make_onehot(buf, seq_length):
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    onehot = [fd[base] for seq in buf for base in seq]
    onehot_np = np.reshape(onehot, (-1, seq_length, 4))
    return onehot_np


def embed(sequence):
    pos = np.random.randint(1, 3)  # choose the insert position
    sequence[pos: pos+9] = 'TCAGCTGAA'


def simulate_data():
    # define a integer --> str dictionary
    letter = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
    # simulate the negative data
    seqlist = []
    for idx in range(1000):
        sequence = np.random.randint(0, 4, 100)
        sequence = ''.join([letter[x] for x in sequence])
        seqlist.append((sequence, 0))
    for idx in range(1000):
        sequence = np.random.randint(0, 4, 100)
        sequence = [letter[x] for x in sequence]
        embed(sequence)
        seqlist.append((''.join(sequence), 1))  # Doing the Join after the embedding for the positive set
    # extracting the sequence data
    dat = np.array(seqlist)[:, 0]
    dat = make_onehot(dat, seq_length=100)
    labels = np.array(seqlist)[:, 1]
    return dat, labels


def build_model():
    """ Define a Keras graph model with sequence and accesibility as input """
    seq_input = Input(shape=(100, 4,), name='seq')
    xs = Conv1D(6, 100, padding="same")(seq_input)
    xs = Activation('relu')(xs)
    # xs = MaxPooling1D(padding="same", strides=15, pool_size=15)(xs)
    # xs = SimpleRNN(32)(seq_input)
    # xs = Flatten()(xs)
    # fully connected dense layers
    xs = LSTM(32)(xs)
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
    model.fit(x=dat, y=labels, epochs=100, batch_size=32)
    # save the model
    model.save('/Users/asheesh/Desktop/Project3/model.hdf5')
    return model


def evaluate_model(model, test_dat, test_labels):
    print test_labels
    probas = model.predict(test_dat)
    auroc = roc_auc_score(test_labels, probas)
    auprc = average_precision_score(test_labels, probas)
    print "auroc:", auroc
    print "auc(pr):", auprc


def ig(model, test_dat, labels):
    att = get_sequence_attribution(model, test_dat)
    behavior = []
    probas = model.predict(test_dat)
    for row, label, probs in zip(att, labels, probas):
        behavior.append((sum(row), int(label), float(probs)))
    np.savetxt("model.behavior.txt", behavior)


def explore_model():
    dat = np.loadtxt("model.behavior.txt")
    print dat
    # Do some graphs!
    # I'd like to figure out average attribution score at all bound/unbound sites.
    bound_zeroatt = 0
    bz = []
    bound_nonzero = 0
    bnz = []
    unbound_zeroatt = 0
    uz = []
    unbound_nonzero = 0
    unz = []
    for attb, status, probas in dat:
        if attb > 0 and status == 1:
            bound_nonzero += 1
            bnz.append(probas)
        elif attb > 0 and status == 0:
            unbound_nonzero += 1
            unz.append(probas)
        elif attb == 0 and status == 1:
            bound_zeroatt += 1
            bz.append(probas)
        else:
            unbound_zeroatt +=1
            uz.append(probas)
    print bound_nonzero, bound_zeroatt, unbound_nonzero, unbound_zeroatt
    # Make a plot from this LATER!
    print np.median(bz)
    print np.median(bnz)
    print np.median(uz)
    print np.median(unz)
    # Even embedding a simple non-probabilistic model, IGs doesn't identify the motif at 75% of sites.
    # The probas are not different for BOUND sites that the IGs can't identify.
    # The probas are different for UNBOUND sites (1%) that IGs does identify
    # Without an LSTM it goes down to about 1/2. (So it's much worse with an LSTM in there!)
    # When I change my input baseline to all zeros, then, without an LSTM it's almost perfect,
    # but with an LSTM it's at 1/2
    # So an LSTM is problem in there!
    # Spot check whether your identification is solid (like you find the motif) in the almost perfect case!
    # Figure out what's (if anything) is different in the sites that the LSTM-model is wrongly interpreting.


def explore_model_more(test_dat, test_labels):
    dat = np.loadtxt("model.behavior.txt")
    print dat
    # Do some graphs!
    # I'd like to figure out average attribution score at all bound/unbound sites.
    bz = []
    bnz = []
    for attb, status, probas in dat:
        if attb > 0 and status == 1:
            bnz.append(probas)
        elif attb == 0 and status == 1:
            bz.append(probas)
    # check the position of the motif!
    # check if there are other k-mers.
    # check if there are different sequence compositions


def main():
    dat, labels = simulate_data()
    X_train, X_test, y_train, y_test = train_test_split(dat, labels)
    # model = fit_model(X_train, y_train.astype(int))
    model = load_model('/Users/asheesh/Desktop/Project3/model.hdf5')
    evaluate_model(model, X_test, y_test.astype(int))
    ig(model, X_test, y_test)
    explore_model()
    # explore_model_more(X_test, y_test)


if __name__ == '__main__':
    main()
    # Current conclusions
    # It DOESN't look like it's systematic! Cause if it was due to something in the sequence, it wouldn't occur 70%
    # of the times.
    # WTF is happening?
