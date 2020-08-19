import numpy as np
import process_data
import sys


def save_batches(file_name, generator, outdir):
    idx = 0
    while idx < 10:
        idx += 1
        X, y, bed_coords = next(generator)
        # np.savetxt('X_' + file_name + str(idx) + '.txt', X, fmt='%s')
        np.savetxt(outdir + 'y_' + file_name + str(idx) + '.txt', y, fmt='%s')
        np.savetxt(outdir + 'coords_' + file_name + str(idx) + '.bed', bed_coords, fmt='%s',
                   delimiter='\t')
    return None


def save_test_set(file_name, test_data, outdir):
    x_test, y_test, coords_test = test_data
    np.savetxt(outdir + 'y_' + file_name + '.txt', y_test, fmt='%s')
    np.savetxt(outdir + 'coords_' + file_name + '.bed', coords_test,
               fmt='%s',
               delimiter='\t')
    return None


def get_train_and_val_generators(genome_sizes, peaks, blacklist, fa):
    filter_chrs_for_training = ['chr10', 'chr18', 'chrUn', 'chrM', 'random']
    tg = process_data.data_generator(genome_sizes_file=genome_sizes,
                                     peaks_file=peaks,
                                     blacklist_file=blacklist,
                                     genome_fasta_file=fa,
                                     window_lenght=500,
                                     batch_size=200,
                                     to_filter=filter_chrs_for_training,
                                     to_keep=None)

    validation_chrs = ['chr10']
    vg = process_data.data_generator(genome_sizes_file=genome_sizes,
                                     peaks_file=peaks,
                                     blacklist_file=blacklist,
                                     genome_fasta_file=fa,
                                     window_lenght=500,
                                     batch_size=500,
                                     to_filter=None,
                                     to_keep=validation_chrs)

    test_chromosome = ['chr18']
    # Testing the class TestSet():
    test_data = process_data.get_test_data(genome_fasta_file=fa,
                                           genome_sizes_file=genome_sizes,
                                           blacklist_file=blacklist,
                                           peaks_file=peaks,
                                           to_keep=test_chromosome,
                                           window_len=500,
                                           stride=500)

    return tg, vg, test_data


