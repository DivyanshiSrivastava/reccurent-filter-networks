"""
Utilities for iterating constructing data sets and iterating over
DNA sequence data.
Pybedtools code from:
https://github.com/uci-cbcl/FactorNet/blob/master/utils.py
Pseudo code structure:
1. Construct a random training set (start with a random negative,
   account for data augmentations later?)
2. Load the data & convert it to onehot. (Look for parallelization tools.)
3. Build a generator

What data does this script take as input or require?
1. The genome sizes file
2. The genome-wide fasta file
3. A blacklist regions file.
4. A ChIP-seq peak file.
"""

import numpy as np
import pandas as pd
import pyfasta
from pybedtools import BedTool

# local imports
import utils


class AccessGenome:
    def __init__(self, genome_fasta_file):
        self.genome_fasta_file = genome_fasta_file

    def get_genome_fasta(self):
        f = pyfasta.Fasta(self.genome_fasta_file)
        return f

    @staticmethod
    def get_onehot_array(seqs, batch_size, window_length):
        """
        Parameters:
            seqs: The sequence array that needs to be converted into one-hot encoded
            features.
            batch_size: mini-batch size
            L: window length
        Returns:
            A one-hot encoded array of shape batch_size * window_len * 4
        """
        onehot_map = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 0, 1],
                      'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        # note: converting all lower-case nucleotides into upper-case here.
        onehot_seqs = [onehot_map[x.upper()] for seq in seqs for x in seq]
        return np.array(onehot_seqs).reshape((batch_size, window_length, 4))

    def get_data_at_coordinates(self, coordinates_df, genome_fasta,
                                window_len, batch_size):
        """
        This method can be used either by:
        1. class ConstructSets: uses this method to return features and labels
           for a training or validation batch.
        2. class ConstructTestData: uses this method to return features and
           labels for the test chromosome co-ordinates and labels.

        Parameters:
            coordinates_df(dataFrame): This method takes as input a Pandas DataFrame with dimensions N * 4
            Where N is the number of samples.
            The columns are: chr, start, stop, label

            genome_fasta (pyFasta npy record): Pyfasta pointer to the fasta file.
            window_len (int): length of windows used for training
            batch_size (int): batch size used for training.

        Returns:
            This method returns a one hot encoded numpy array (X) and a np
            vector y.
            Both X and y are numpy arrays.
            X shape: (batch size, L, 4)
            y shape: (batch size,)
        """
        batch_y = coordinates_df['label']
        batch_X = []
        for chrom, start, stop, y in coordinates_df.values:
            batch_X.append(genome_fasta[chrom][int(start):int(stop)])
        # converting this data into onehot
        batch_X_onehot = AccessGenome.get_onehot_array(batch_X,
                                                       window_length=window_len,
                                                       batch_size=batch_size)
        return batch_X_onehot, batch_y.values


class ConstructSets(AccessGenome):
    """
    Notes:
        chip_coords is the filtered chip_seq file, it either contains only
        train chromosomes or validation chromosomes based on the input.
    """

    def __init__(self, genome_sizes_file, genome_fasta_file, blacklist_file,
                 chip_coords, window_length, exclusion_btd_obj,
                 curr_genome_bed, batch_size):
        super().__init__(genome_fasta_file)
        self.genome_sizes_file = genome_sizes_file
        self.blacklist_file = blacklist_file
        self.chip_coords = chip_coords
        self.L = window_length
        self.exclusion_bdt_obj = exclusion_btd_obj
        self.curr_genome_bed = curr_genome_bed
        self.batch_size = batch_size

    def apply_random_shift(self, coords):
        """
        This function takes as input a set of bed co-ordinates
        It finds the mid-point for each record or Interval in the bed file,
        shifts the mid-point, and generates a window of
        length self.L.

        Calculating the shift:

        For each interval, find the mid-point.
        In this case, multiGPS is outputting 1 bp windows,
        so just taking the "start" as the mid-point.
        For example:

        Asc1.bed record:
        chr18   71940632   71940633
        mid-point: 71940632

        If training window length is L, then we must ensure that the
        peak center is still within the training window.
        Therefore: -L/2 < shift < L/2
        To add in a buffer: -L/2 + 25 <= shift <= L/2 + 25
        # Note: The 50 here is a tunable hyper-parameter.

        Parameters:
            coords(pandas dataFrame): This is an input bedfile
        Returns:
            shifted_coords(pandas dataFrame): The output bedfile with shifted coords
        """
        # defining the random shift
        low = int(-self.L/2 + 25)
        high = int(self.L/2 - 25)
        coords['random_shift'] = np.random.randint(low=low, high=high,
                                                   size=len(coords))
        coords['s_start'] = coords['start'] + coords['random_shift'] - int(self.L/2)
        coords['s_end'] = coords['start'] + coords['random_shift'] + int(self.L/2)

        # making a new dataFrame containing the new shifted coords.
        shifted_coords = coords[['chr', 's_start', 's_end']]
        shifted_coords.columns = ['chr', 'start', 'end']

        return shifted_coords

    def define_coordinates(self):
        """
        Use the chip-seq peak file and the blacklist files to define a bound
        set and an unbound set of sites. The ratio of bound to unbound is 1:2,
        but can be controlled using the parameter "ratio".

        The unbound/negative set is chosen randomly from the genome.
        """
        positive_sample_size = int(self.batch_size/2)

        # taking a sample from the chip_coords file,
        # i.e. sub-setting 50 rows from self.chip_coords
        positive_sample = self.chip_coords.sample(n=positive_sample_size)
        # taking only the first three columns
        # (removing multiGPS scores & caller names)
        positive_sample = positive_sample.iloc[:, 0:3]
        # applying a random shift that returns 200 bp windows.
        positive_sample_w_shift = self.apply_random_shift(positive_sample)
        # creating a BedTool object for further use:
        positive_sample_bdt_obj = BedTool.from_dataframe(positive_sample_w_shift)

        negative_sample_bdt_obj = BedTool.shuffle(positive_sample_bdt_obj,
                                                  g=self.genome_sizes_file,
                                                  incl=self.curr_genome_bed.fn,
                                                  excl=self.exclusion_bdt_obj.fn)
        negative_sample = negative_sample_bdt_obj.to_dataframe()
        negative_sample.columns = ['chr', 'start', 'end'] # naming such that the
        # column names are consistent with positive_samples

        # adding in labels:
        positive_sample_w_shift['label'] = 1
        negative_sample['label'] = 0

        # mixing and shuffling positive and negative set:
        training_coords = pd.concat([positive_sample_w_shift, negative_sample])
        # randomly shuffle the dataFrame
        training_coords = training_coords.sample(frac=1)
        return training_coords

    def get_data(self):
        # get mini-batch co-ordinates:
        coords_for_data = self.define_coordinates()
        # get the fasta file:
        genome_fasta = super(ConstructSets, self).get_genome_fasta()
        dat_X, labels_y = super().get_data_at_coordinates(coordinates_df=coords_for_data,
                                                          genome_fasta=genome_fasta,
                                                          window_len=self.L,
                                                          batch_size=self.batch_size)
        return dat_X, labels_y, coords_for_data


class TestSet(AccessGenome):

    def __init__(self, genome_fasta_file, genome_sizes_file, peaks_file,
                 blacklist_file, window_len, stride):
        super().__init__(genome_fasta_file)
        self.genome_sizes_file = genome_sizes_file
        self.peaks_file = peaks_file
        self.blacklist_file = blacklist_file
        self.window_len = window_len
        self.stride = stride

    def define_coordinates(self):
        """
        This function loads and returns coords & labels for the test set.

        Logic for assigning test set labels:
        The multiGPS peak files are used as inputs; and expanded to record
        25 bp windows around the peak center.
        if 100% of peak center lies in window:
            label bound.
        elif < 100% of peak center lies in the window:
            label ambiguous.
        else:
            label unbound.

        Returns:
            test_coords (pd dataFrame): A dataFrame with chr, start, end and
            labels
        """
        genome_sizes = pd.read_csv(self.genome_sizes_file, sep="\t",
                                   names=['chr', 'len'])
        # subset the test chromosome:
        genome_test = genome_sizes[genome_sizes['chr'] == 'chr10']
        end_idx = genome_test.iloc[0, 1]
        chromosome = genome_test.iloc[0, 0]
        test_set = []
        start_idx = 0
        while start_idx + self.window_len < end_idx:
            curr_interval = [chromosome, start_idx, start_idx + self.window_len]
            start_idx += self.stride
            test_set.append(curr_interval)

        test_df = pd.DataFrame(test_set, columns=['chr', 'start', 'stop'])
        test_bdt_obj = BedTool.from_dataframe(test_df)

        chip_peaks = utils.load_chipseq_data(chip_peaks_file=self.peaks_file,
                                             to_keep=['chr10'])
        # note: multiGPS reports 1 bp separated start and end,
        # centered on the ChIP-seq peak.
        chip_peaks['start'] = chip_peaks['start'] - 12
        chip_peaks['end'] = chip_peaks['end'] + 12

        chip_peaks = chip_peaks[['chr', 'start', 'end']]
        chip_peaks_bdt_obj = BedTool.from_dataframe(chip_peaks)

        blacklist_exclusion_windows = BedTool(self.blacklist_file)
        # intersecting
        unbound_data = test_bdt_obj.intersect(chip_peaks_bdt_obj, v=True)
        unbound_data = unbound_data.intersect(blacklist_exclusion_windows,
                                              v=True)
        # i.e. if there is any overlap with chip_peaks, that window is not
        # reported
        bound_data = test_bdt_obj.intersect(chip_peaks_bdt_obj, F=1, u=True)
        bound_data = bound_data.intersect(blacklist_exclusion_windows,
                                          v=True)  # removing blacklist windows
        # i.e. the entire 25 bp window should be in the 200 bp test_bdt_obj \
        # window
        # making data-frames
        bound_data_df = bound_data.to_dataframe()
        bound_data_df['label'] = 1
        unbound_data_df = unbound_data.to_dataframe()
        unbound_data_df['label'] = 0
        # exiting
        test_coords = pd.concat([bound_data_df, unbound_data_df])
        return test_coords

    def get_data(self):
        # get mini-batch co-ordinates:
        coords_for_data = self.define_coordinates()
        # get the fasta file:
        genome_fasta = super().get_genome_fasta()
        data_size = len(coords_for_data)
        dat_X, labels_y = super().get_data_at_coordinates(
            coordinates_df=coords_for_data,
            genome_fasta=genome_fasta,
            window_len=self.window_len,
            batch_size=data_size)
        return dat_X, labels_y, coords_for_data


def data_generator(genome_sizes_file, peaks_file, genome_fasta_file,
                   blacklist_file, to_keep, to_filter):
    """
    This generator can either generate training data or validation data based on
    the to_keep and to_filter arguments.

    The train generate uses the to_filter argument, whereas to_keep=None
    For example:
    train_generator:  to_filter=['chr10', 'chr17, 'chrUn', 'chrM', 'random']
    i.e. In this construction; chr10 and chr17 can be used for testing/validation.

    The val generator uses the to_keep argument, whereas to_filter=None.
    For example:
    val_generator: to_keep=['chr17']
    i.e. In this construction; chr17 data is used for validation.

    Additional Parameters:
        genome_sizes_file: sizes
        peaks_file: multiGPS formatted BED file
        blacklist_file: BED format blacklist file
        genome_fasta_file: fasta file for the whole genome
        batch_size (int): batch size used for training and validation batches
        window_len (int): the length of windows used for training and testing.
    """
    # load the genome_sizes_file:
    genome_bed_val = utils.get_genome_sizes(genome_sizes_file, to_keep=to_keep,
                                            to_filter=to_filter)
    # loading the chip-seq bed file
    chip_seq_coordinates = utils.load_chipseq_data(peaks_file,
                                                   to_keep=to_keep,
                                                   to_filter=to_filter)
    # loading the exclusion coords:
    exclusion_windows_bdt = utils.exclusion_regions(blacklist_file,
                                                    chip_seq_coordinates)
    # constructing the training set
    construct_val_sets = ConstructSets(genome_sizes_file=genome_sizes_file,
                                       genome_fasta_file=genome_fasta_file,
                                       blacklist_file=blacklist_file,
                                       chip_coords=chip_seq_coordinates,
                                       exclusion_btd_obj=exclusion_windows_bdt,
                                       window_length=200,
                                       curr_genome_bed=genome_bed_val,
                                       batch_size=100)
    while True:
        X_val, y_val, coords = construct_val_sets.get_data()
        yield X_val, y_val, coords


def get_test_data(genome_sizes_file, peaks_file, genome_fasta_file,
                  blacklist_file):
    ts = TestSet(genome_fasta_file=genome_fasta_file, genome_sizes_file=genome_sizes_file,
                 peaks_file=peaks_file, blacklist_file=blacklist_file,
                 window_len=500, stride=100)
    X_test, y_test, coords = ts.get_data()
    return X_test, y_test, coords

