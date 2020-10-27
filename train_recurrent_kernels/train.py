"""
Train models that predict TF binding using either CNNs or
kernels with recurrent filters.
Author: Divyanshi Srivastava (dvs5680@psu.edu)
The Pennsylvania State University
"""
import argparse
from train_recurrent_kernels import convnet, rf_net
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the processing pipeline')
    parser.add_argument('genome_sizes', help='Input genome sizes file')
    parser.add_argument('blacklist', help='Input blacklist file in BED format')
    parser.add_argument('fa', help='Input genomic fasta file')
    parser.add_argument('peaks', help='Input ChIP-seq peaks file in multiGPS format')
    parser.add_argument('results_dir', help='Directory for storing results')
    # specify defaults here
    parser.add_argument('--params_yaml', help='Hyper-parameters_yaml_file')
    parser.add_argument('--acc_regions_file', help='BED file:accessible regions')
    parser.add_argument('--network_type', help='Either CNN or RFN',
                        default='CNN')

    args = parser.parse_args()

    with open(args.params_yaml, 'r') as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    batch_size = params['parameters']['batch_size']
    patience = params['parameters']['patience']
    ratio = params['parameters']['ratio']
    lr = params['parameters']['lr']
    optimizer = params['parameters']['optimizer']
    steps_per_epoch = params['parameters']['steps']

    filter_width = params['parameters']['filter_width']
    no_of_filters = params['parameters']['no_of_filters']

    if args.network_type == 'CNN':
        convnet.train_model(genome_size=args.genome_sizes, peaks=args.peaks,
                            blacklist=args.blacklist, fa=args.fa,
                            results_dir=args.results_dir, batch_size=batch_size,
                            steps=steps_per_epoch, patience=patience,
                            acc_regions_file=args.acc_regions_file,
                            learning_rate=lr, opt=optimizer, ratios=ratio,
                            filter_width=filter_width,
                            no_of_filters=no_of_filters)
    else:
        rf_net.train_model(genome_size=args.genome_sizes, peaks=args.peaks,
                           blacklist=args.blacklist, fa=args.fa,
                           results_dir=args.results_dir, batch_size=batch_size,
                           steps=steps_per_epoch, patience=patience,
                           acc_regions_file=args.acc_regions_file,
                           learning_rate=lr, opt=optimizer, ratios=ratio,
                           filter_width=filter_width,
                           no_of_filters=no_of_filters)
