import numpy as np
from simulatedata import make_onehot

# keras imports
from keras.models import load_model


def get_filters(model):
    filter_list = []
    for layer in model.layers:
        if layer.name == 'convolution_1':
            # This is my layer of interest
            print layer.name
            weights = layer.get_weights()
            # These are the weights here.
            W, b = weights
            print W.shape
            # Note: The original architecture has 64 filters.
            for filter in range(64):
                filter_weights = W[:, :, filter]
                filter_list.append(filter_weights)
    return filter_list


def make_motif_onehot(motif):
    fd = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    onehot = [fd[base] for seq in motif for base in seq]
    onehot_np = np.reshape(onehot, (8, 4))
    return onehot_np


def element_wise_product(mat1, mat2):
    return np.sum(mat1 * mat2)


def convolve(filter, motif):
    len_mat_filter, depth = filter.shape
    len_mat_motif, depth = motif.shape

    # Assumption: Motif is smaller than the filter size
    assert depth == 4  # I'm dealing with sequences
    assert len_mat_filter > len_mat_motif  # The filter is larger than the motif

    position = len_mat_motif
    start_idx = 0
    scores = []

    while position <= len_mat_filter:
        # Note: convolution score is a scalar
        convolution_score = element_wise_product(filter[start_idx:position], motif)
        scores.append(convolution_score)
        start_idx += 1
        position += 1
    return max(scores)


if __name__ == "__main__":
    model = load_model('/Users/divyanshisrivastava/Desktop/model.hdf5')
    get_filters(model)

    # Define the embedded motif
    motif = 'TGATTTAT'
    motif = 'AAAAAAAA'
    motif_onehot = make_motif_onehot(motif)

    filter_list = get_filters(model)

    scores = []
    for filter_weights in filter_list:
        score = convolve(filter_weights, motif_onehot)
        scores.append(score)
    print np.sort(scores)
    print max(scores)



