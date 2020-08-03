import numpy as np
import process_data
import argparse


parser = argparse.ArgumentParser(description='Test the processing pipeline')
parser.add_argument('genome_sizes', help='Input genome sizes file')
parser.add_argument('blacklist', help='Input blacklist file in BED format')
parser.add_argument('fa', help='Input genomic fasta file')
# Here, I should potentially accept a BED file; check for that.
parser.add_argument('peaks', help='Input ChIP-seq peaks file in multiGPS format')

args = parser.parse_args()


def save_batches(file_name, generator):
    idx = 0
    while idx <= 10:
        idx += 1
        X, y, bed_coords = next(generator)
        np.savetxt('X_' + file_name + str(idx) + '.txt', X, fmt='%s')
        np.savetxt('y_' + file_name + str(idx) + '.txt', y, fmt='%s')
        np.savetxt('coords_' + file_name + str(idx) + '.bed', y, fmt='%s',
                   delimiter='\t')
    return None


tg = process_data.train_generator(genome_sizes_file=args.genome_sizes,
                                  peaks_file=args.peaks,
                                  blacklist_file=args.blacklist,
                                  genome_fasta_file=args.fa)

vg = process_data.val_generator(genome_sizes_file=args.genome_sizes,
                                peaks_file=args.peaks,
                                blacklist_file=args.blacklist,
                                genome_fasta_file=args.fa)

# Iterate 10 times with both the train & val generators; save batches to disk.
save_batches('train', tg)
save_batches('validation', vg)


