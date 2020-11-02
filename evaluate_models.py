"""
This stand-alone script evaluates ANY model which is passed as an argument.
This script uses the evaluation functionalities
in process_data.py,  rf_net.py and conv_net.py
Note: This is not run as part of the CL usage of the train.py.
Instead, internal evaluation scripts defined in get_data and rf_net
are used.
"""

import sys
import subprocess
import numpy as np
import pandas as pd
from process_data import get_test_data
import utils
from tensorflow.keras.models import load_model
import rf_net


def get_probabilities(model, test_data_tuple, outdir):
    """
    Saves and returns model probabilities at the test set.
    Parameters:
        model: A Keras Model
        test_data_tuple: A tuple of X, y and bed co-ords for X
        outdir: Output directory path
    Return:
         A list of probabilities
    """
    x_test, y_test, bed_coords_test = test_data_tuple
    x_test_bound = x_test[y_test == 1]
    print(x_test_bound.shape)
    model_probas = model.predict(x_test_bound)
    subprocess.call(['mkdir', outdir])
    probas_file = outdir + '.bound_probas.txt'
    bed_coords_file = outdir + 'coordinates.test.bed'
    np.savetxt(probas_file, model_probas, fmt='%1.2f')
    np.savetxt(bed_coords_file, bed_coords_test, fmt='%s', delimiter='\t')
    return model_probas


if __name__ == "__main__":
    fa = sys.argv[1]
    genome_sizes = sys.argv[2]
    blacklist_file = sys.argv[3]
    peaks_file = sys.argv[4]
    model_path = sys.argv[5]
    out_dir = sys.argv[6]

    # Note: to_keep must be a list of chromosomes.
    # Example: to_keep = ['chr10']

    test_data = get_test_data(genome_fasta_file=fa, peaks_file=peaks_file,
                              genome_sizes_file=genome_sizes,
                              blacklist_file=blacklist_file,
                              to_keep=['chr10'], window_len=500, stride=500)

    model = load_model(model_path,
                       custom_objects={'DistributeInputLayer':
                                        rf_net.DistributeInputLayer})
    X_test, y_test, bed = test_data
    probas_bound = get_probabilities(model, test_data, out_dir)
