import argparse
import convnet
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the processing pipeline')
    parser.add_argument('genome_sizes', help='Input genome sizes file')
    parser.add_argument('blacklist', help='Input blacklist file in BED format')
    parser.add_argument('fa', help='Input genomic fasta file')
    parser.add_argument('peaks', help='Input ChIP-seq peaks file in multiGPS format')
    parser.add_argument('results_dir', help='Directory for storing results')
    parser.add_argument('--params_yaml', help='Hyper-parameters_yaml_file')
    parser.add_argument('--acc_regions_file', help='BED file:accessible regions')

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

    convnet.train_model(genome_size=args.genome_sizes, peaks=args.peaks,
                        blacklist=args.blacklist, fa=args.fa,
                        results_dir=args.results_dir, batch_size=batch_size,
                        steps=steps_per_epoch, patience=patience,
                        acc_regions_file=args.acc_regions_file,
                        learning_rate=lr, opt=optimizer, ratios=ratio)
