import argparse
import convnet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the processing pipeline')
    parser.add_argument('genome_sizes', help='Input genome sizes file')
    parser.add_argument('blacklist', help='Input blacklist file in BED format')
    parser.add_argument('fa', help='Input genomic fasta file')
    parser.add_argument('peaks', help='Input ChIP-seq peaks file in multiGPS format')
    parser.add_argument('results_dir', help='Directory for storing results')
    parser.add_argument('--batch_size', help='batch_size')
    parser.add_argument('--steps_per_epoch', help='steps_per_epoch')
    parser.add_argument('--patience', help='patience')

    args = parser.parse_args()

    # note: all parameters are specified in train_model
    convnet.train_model(genome_size=args.genome_sizes, peaks=args.peaks,
                        blacklist=args.blacklist, fa=args.fa,
                        results_dir=args.results_dir, batch_size=args.batch_size,
                        steps=args.steps_per_epoch, patience=args.patience)

