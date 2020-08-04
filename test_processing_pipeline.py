import numpy as np
import process_data
import argparse


parser = argparse.ArgumentParser(description='Test the processing pipeline')
parser.add_argument('genome_sizes', help='Input genome sizes file')
parser.add_argument('blacklist', help='Input blacklist file in BED format')
parser.add_argument('fa', help='Input genomic fasta file')
parser.add_argument('peaks', help='Input ChIP-seq peaks file in multiGPS format')
parser.add_argument('outdir', help='Directory for storing temp testing files')
args = parser.parse_args()


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


filter_chrs_for_training = ['chr10', 'chr17', 'chrUn', 'chrM', 'random']
tg = process_data.data_generator(genome_sizes_file=args.genome_sizes,
                                 peaks_file=args.peaks,
                                 blacklist_file=args.blacklist,
                                 genome_fasta_file=args.fa,
                                 window_lenght=500,
                                 batch_size=100,
                                 to_filter=filter_chrs_for_training,
                                 to_keep=None)

validation_chrs = ['chr17']
vg = process_data.data_generator(genome_sizes_file=args.genome_sizes,
                                 peaks_file=args.peaks,
                                 blacklist_file=args.blacklist,
                                 genome_fasta_file=args.fa,
                                 window_lenght=500,
                                 batch_size=100,
                                 to_filter=None,
                                 to_keep=validation_chrs)

# Iterate N times with both the train & val generators; save batches to disk.
# save_batches('train', tg, outdir=args.outdir)
# save_batches('validation', vg, outdir=args.outdir)

test_chromosome = ['chr10']
# Testing the class TestSet():
test_data = process_data.get_test_data(genome_fasta_file=args.fa,
                                       genome_sizes_file=args.genome_sizes,
                                       blacklist_file=args.blacklist,
                                       peaks_file=args.peaks,
                                       to_keep=test_chromosome,
                                       window_len=500,
                                       stride=500)
save_test_set('test', test_data=test_data, outdir=args.outdir)







