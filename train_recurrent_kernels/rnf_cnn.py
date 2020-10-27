"""
Apply recurrent neural filters to DNA sequence data.
This implementation uses Tensorflow and Keras.
"""

import numpy as np
import keras
import argparse

# sk-learn imports
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
from keras.engine import Layer
import keras.backend as K

# local imports
from simulatedata import TrainingData, TestData


def make_onehot(buf, seq_length):
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
          'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1],
          'N': [0, 0, 0, 0]}
    one_hot = [fd[base] for seq in buf for base in seq]
    one_hot_np = np.reshape(one_hot, (-1, seq_length, 4))
    return one_hot_np


# This class is adapted from:
# https://github.com/bloomberg/cnn-rnf/blob/master/cnn_keras.py
class DistInputLayer(Layer):
    """
        Distribute word vectors into chunks - input for the convolution operation
        Input dim: [batch_size x sentence_len x word_vec_dim]
        Output dim: [batch_size x (sentence_len - filter_width + 1) x filter_width x word_vec_dim]
    """

    def __init__(self, filter_width, seq_len, **kwargs):
        super(DistInputLayer, self).__init__(**kwargs)
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


class RnfFast:
    def __init__(self, seq_length, rnf_filters, rnf_kernel_size, conv_filters, conv_kernel_size,
                 dense_nodes, rnf_dim):
        self.seq_length = seq_length
        self.rnf_kernel_size = rnf_kernel_size
        self.rnf_filters = rnf_filters
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.dense_nodes = dense_nodes
        self.rnf_dim = rnf_dim

    def rnf_model(self):

        seq_input = Input(shape=(self.seq_length, 4,), name='seq')
        embedded_inp = DistInputLayer(filter_width=self.rnf_kernel_size, seq_len=self.seq_length)(seq_input)
        # Shape:(?, L-F+1, F, D)
        # The TimeDistributed Layer treats index 1 in this input as \
        # independent time steps.
        # So here, the RNN is being applied to every chunk,
        xs = TimeDistributed(LSTM(self.rnf_dim))(embedded_inp)
        xs = Activation('relu')(xs)
        print(xs.shape)
        # Shape:(?, L-F+1, RNF_DIM) # Note here, the LSTM is producing
        # a single output with dimension RNF_DIM
        # Include an L-1 norm at the subsequent dense layer?
        xs = Conv1D(filters=self.conv_filters, kernel_size=self.conv_kernel_size)(xs)
        xs = Activation('relu')(xs)
        # From here on, the model is identical to the convolutional model
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


def fit_model(dat, labels, batch_size, seq_length, mp):
    # choose the parameters here:
    conv_filters = [32, 64, 128, 256]
    conv_kernel_size = [6, 12, 16, 24]
    dense_nodes = [32, 64, 128]
    rnf_filters = [6, 12, 24]
    rnf_kernel_size = [6, 12, 16, 24]
    rnf_dim = [1, 2, 4]
    params = []
    for parameters in [conv_filters, conv_kernel_size, dense_nodes, rnf_filters, rnf_kernel_size, rnf_dim]:
        options = len(parameters)
        rnum = np.random.choice(options)
        params.append(parameters[rnum])
    architecture = RnfFast(seq_length=seq_length, conv_filters=params[0], conv_kernel_size=params[1],
                           dense_nodes=params[2], rnf_filters=params[3], rnf_kernel_size=params[4],
                           rnf_dim=params[5])
    model = architecture.rnf_model()

    # Splitting data into test and train
    X_train, X_test, y_train, y_test = train_test_split(dat, labels.astype(int))
    # Define the optimization here:
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', mode='min',
                              verbose=1, patience=8)
    model.fit(x=X_train, y=y_train, epochs=32, batch_size=batch_size,
              validation_split=0.2, callbacks=[earlystop])

    probas = model.predict(X_test)
    auroc = roc_auc_score(y_test, probas)
    auprc = average_precision_score(y_test, probas)
    return model, params, auroc, auprc


def main():

    parser = argparse.ArgumentParser(description='Compare CNNs to RNFs for DNA sequence')
    parser.add_argument('design', help='Tab-delimited file with the 4 motifs to use')
    parser.add_argument('outfile', help='Outfile with model deets')
    parser.add_argument('mtype', help='conv or rnf model')
    parser.add_argument('nseqs', help='N of seqs to test on')
    parser.add_argument('nseqs_multiplier', help='N of seqs to test on')
    parser.add_argument('nseqs_negative', help='N of seqs to test on')
    parser.add_argument('idx', help='curr_idx')
    parser.add_argument('MP_status', help='To max pool after 1 conv. layer or not. ')

    args = parser.parse_args()

    motif_file = args.design
    motif_a, motif_b, motif_flipped, motif_random = np.loadtxt(motif_file, dtype=str, delimiter='\t')
    # Instantiate an instance of TrainingData
    # Change N / Test over many N's.
    train_data = TrainingData(motif_a=motif_a, motif_b=motif_b, N=int(args.nseqs),
                              N_mult=int(args.nseqs_multiplier), N_neg=int(args.nseqs_negative),
                              seq_length=500)

    # Getting the synthetic X and y data
    dat, labels = train_data.simulate_data()
    model, params, auroc, auprc = fit_model(dat=dat, labels=labels, batch_size=64,
                                            seq_length=500, mp=args.MP_status)

    # Instantiating an instance of Evaluating Composition
    td = TestData(seq_length=500, model=model)
    # Testing across 4 motifs

    outfile = args.outfile + '.' + args.mtype + '.' + args.nseqs + '.' + args.nseqs_multiplier\
              + '.' + args.nseqs_negative + '.' + args.MP_status + '.' + args.idx + '.txt'

    with open(outfile, 'w') as fp_out:
        # Write to file params and dtype and performance
        fp_out.write('auROC:{}\n'.format(auroc))
        fp_out.write('auPRC:{}\n'.format(auprc))
        fp_out.write('Model:{}\n'.format(args.mtype))
        fp_out.write('Params:{}\n'.format(params))
        # Perf.
        motif_a_score = td.simulate_test_dat(motif_a)
        fp_out.write('{},{}\n'.format(motif_a, motif_a_score))

        motif_b_score = td.simulate_test_dat(motif_b)
        fp_out.write('{},{}\n'.format(motif_b, motif_b_score))

        motif_flipped_score = td.simulate_test_dat(motif_flipped)
        fp_out.write('{},{}\n'.format(motif_flipped, motif_flipped_score))

        motif_random_score = td.simulate_test_dat(motif_random)
        fp_out.write('{},{}\n'.format(motif_random, motif_random_score))
    model.save(outfile + '.hdf5')


if __name__ == '__main__':
    main()