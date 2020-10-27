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
import pybedtools
from pybedtools import BedTool

# local imports
import utils

pybedtools.set_tempdir('/storage/home/dvs5680/scratch/')
np.random.seed(9)


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
        onehot_map = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0],
                      'C': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
        # note: converting all lower-case nucleotides into upper-case here.
        onehot_seqs = [onehot_map[x.upper()] for seq in seqs for x in seq]
        onehot_data = np.reshape(onehot_seqs, newshape=(batch_size, window_length, 4))
        # remove the reshaping step:
        # onehot_data = list()
        # for sequence in seqs:
        #     onehot_seq = list()
        #     for nucleotide in sequence:
        #         onehot_seq.append(onehot_map[nucleotide.upper()])
        #     onehot_data.append(onehot_seq)
        return onehot_data

    def rev_comp(self, inp_str):
        rc_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'c': 'g',
                   'g': 'c', 't': 'a', 'a': 't', 'n': 'n', 'N': 'N'}
        outp_str = list()
        for nucl in inp_str:
            outp_str.append(rc_dict[nucl])
        return ''.join(outp_str)

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
        seq_len = []

        batch_size = len(batch_y)
        idx = 0
        for chrom, start, stop, y in coordinates_df.values:
            fa_seq = genome_fasta[chrom][int(start):int(stop)]
            # Adding reverse complements into the training process:
            if idx <= int(batch_size/2):
                batch_X.append(fa_seq)
            else:
                batch_X.append(self.rev_comp(fa_seq))
            idx += 1
            seq_len.append(len(fa_seq))
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
                 chip_coords, window_length, exclusion_df,
                 curr_genome_bed, batch_size, acc_regions_file, flanks, ratios):
        super().__init__(genome_fasta_file)
        self.genome_sizes_file = genome_sizes_file
        self.blacklist_file = blacklist_file
        self.chip_coords = chip_coords
        self.L = window_length
        self.exclusion_df = exclusion_df  # This is df, convert to a bdt object.
        self.curr_genome_bed = curr_genome_bed
        # self.curr_genome_bed is is a df, convert to a bdt obj.
        self.batch_size = batch_size
        self.acc_regions_file = acc_regions_file
        self.flanks_df = flanks  # This is df, convert to a bdt object.
        self.ratios = ratios  # list

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
        shifted_coords = coords.loc[:, ('chr', 's_start', 's_end')]
        shifted_coords.columns = ['chr', 'start', 'end']

        return shifted_coords

    def define_coordinates(self):
        """
        Use the chip-seq peak file and the blacklist files to define a bound
        set and an unbound set of sites. The ratio of bound to unbound is 1:N,
        but can be controlled using the parameter "ratio".

        The unbound/negative set is chosen randomly from the genome.(ha)
        """
        # positive sample
        # Take a sample from the chip_coords file,
        # Then apply a random shift that returns 500 bp windows.
        # Create a BedTool object for further use.
        bound_sample_size = int(self.batch_size)
        bound_sample = self.chip_coords.sample(n=bound_sample_size)
        bound_sample_w_shift = self.apply_random_shift(bound_sample)
        bound_sample_bdt_obj = BedTool.from_dataframe(bound_sample_w_shift)
        bound_sample_w_shift['label'] = 1
        # negative samples: random
        # note: the self.curr_genome_bed.fn contains only training chromosomes.
        # Creates a DF.
        curr_genome_bdt = BedTool.from_dataframe(self.curr_genome_bed)
        exclusion_bdt_obj = BedTool.from_dataframe(self.exclusion_df)
        unbound_random_bdt_obj = BedTool.shuffle(bound_sample_bdt_obj,
                                                 g=self.genome_sizes_file,
                                                 incl=curr_genome_bdt.fn,
                                                 excl=exclusion_bdt_obj.fn)
        unbound_random_df = unbound_random_bdt_obj.to_dataframe()
        unbound_random_df.columns = ['chr', 'start', 'end']
        unbound_random_df['label'] = 0
        # negative sample: flanking windows
        flanks_bdt = BedTool.from_dataframe(self.flanks_df)
        unbound_flanks_bdt_obj = flanks_bdt.intersect(curr_genome_bdt)
        unbound_flanks_df = unbound_flanks_bdt_obj.to_dataframe()
        unbound_flanks_df.columns = ['chr', 'start', 'end']
        unbound_flanks_df['label'] = 0
        unbound_flanks_df = unbound_flanks_df.sample(frac=1)
        # negative sample: pre-accessible/accessible
        # get accessibility domains.
        # Use BedTools shuffle to place windows in these regions.
        # regions_acc_bdt_obj = BedTool(self.acc_regions_file)
        # regions_acc_bdt_obj = regions_acc_bdt_obj.intersect(self.curr_genome_bed.fn)
        # negative samples/pre-accessible
        # unbound_acc_bdt_obj = BedTool.shuffle(bound_sample_bdt_obj,
        #                                       g=self.genome_sizes_file,
        #                                       incl=regions_acc_bdt_obj.fn,
        #                                       excl=self.exclusion_bdt_obj.fn)
        # unbound_acc_df = unbound_acc_bdt_obj.to_dataframe()
        # unbound_acc_df.columns = ['chr', 'start', 'end']
        # unbound_acc_df['label'] = 0

        # Training set based on the training ratios:
        ratios = self.ratios  # tuple or list
        # example: (1, 2, 4, 1)
        denom = np.sum(self.ratios)
        split = [int((frac/denom) * self.batch_size) for frac in self.ratios]
        b_r, ub_rand, ub_flanks = split
        training_coords = pd.concat([bound_sample_w_shift[0: b_r],
                                     unbound_random_df[b_r: (b_r + ub_rand)],
                                     unbound_flanks_df[(b_r + ub_rand): self.batch_size]])

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
                 blacklist_file, window_len, stride, to_keep):
        super().__init__(genome_fasta_file)
        self.genome_sizes_file = genome_sizes_file
        self.peaks_file = peaks_file
        self.blacklist_file = blacklist_file
        self.window_len = window_len
        self.stride = stride
        self.to_keep = to_keep

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
        genome_test = genome_sizes[genome_sizes['chr'] == self.to_keep[0]]
        # the assumption here is that to_keep is a single chromosome list.
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
                                             to_keep=self.to_keep,
                                             genome_sizes_file=self.genome_sizes_file)
        # note: multiGPS reports 1 bp separated start and end,
        # centered on the ChIP-seq peak.
        chip_peaks['start'] = chip_peaks['start'] - int(self.window_len/2)
        # (i.e. 250 if window_len=500 )
        chip_peaks['end'] = chip_peaks['end'] + int(self.window_len/2 - 1)
        # (i.e. 249 if window_len=500); multiGPS reports 1bp intervals

        chip_peaks = chip_peaks[['chr', 'start', 'end']]
        chip_peaks_bdt_obj = BedTool.from_dataframe(chip_peaks)

        blacklist_exclusion_windows = BedTool(self.blacklist_file)
        # intersecting
        unbound_data = test_bdt_obj.intersect(chip_peaks_bdt_obj, v=True)
        unbound_data = unbound_data.intersect(blacklist_exclusion_windows,
                                              v=True)
        # i.e. if there is any overlap with chip_peaks, that window is not
        # reported
        # removing blacklist windows
        bound_data = chip_peaks_bdt_obj.intersect(blacklist_exclusion_windows,
                                                  v=True)
        # i.e. the entire 500 bp window is the positive window.
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
                   blacklist_file, to_keep, to_filter,
                   window_lenght, batch_size, acc_regions_file,
                   ratios):
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
        peaks_file: multiGPS formatted *events* file
        blacklist_file: BED format blacklist file
        genome_fasta_file: fasta file for the whole genome
        batch_size (int): batch size used for training and validation batches
        window_len (int): the length of windows used for training and testing.
    """
    # load the genome_sizes_file:
    genome_bed_val = utils.get_genome_sizes(genome_sizes_file, to_keep=to_keep,
                                            to_filter=to_filter)
    genome_bed_df = genome_bed_val.to_dataframe()
    # loading the chip-seq bed file
    chip_seq_coordinates = utils.load_chipseq_data(peaks_file,
                                                   genome_sizes_file=genome_sizes_file,
                                                   to_keep=to_keep,
                                                   to_filter=to_filter)

    def make_flanks(lower_lim, upper_lim):
        # getting a list of chip-seq flanking windows:
        # (can be a separate fn in utils)
        flanks_left = chip_seq_coordinates.copy()
        flanks_right = chip_seq_coordinates.copy()
        flanks_left['start'] = chip_seq_coordinates['start'] - upper_lim
        flanks_left['end'] = chip_seq_coordinates['start'] - lower_lim
        flanks_right['start'] = chip_seq_coordinates['start'] + lower_lim
        flanks_right['end'] = chip_seq_coordinates['start'] + upper_lim
        return flanks_left, flanks_right

    fl_r, fl_l = make_flanks(lower_lim=250, upper_lim=750)
    fl_r_2, fl_l_2 = make_flanks(lower_lim=200, upper_lim=700)
    fl_r_3, fl_l_3 = make_flanks(lower_lim=1500, upper_lim=2000)
    fl_r_4, fl_l_4 = make_flanks(lower_lim=1000, upper_lim=1500)
    flanks = pd.concat([fl_r, fl_l, fl_r_2, fl_l_2, fl_l_3, fl_r_3, fl_r_4, fl_l_4])
    # flanks_bdt_obj = BedTool.from_dataframe(flanks)
    # converting the df to a bedtools object inside the generator, to enable a
    # py-bedtools cleanup otherwise.
    # print(flanks_bdt_obj.head())
    # flanks_bdt_obj = flanks_bdt_obj.intersect(BedTool.from_dataframe(chip_seq_coordinates),
    #                                           v=True)
    # print(flanks_bdt_obj.head)

    # loading the exclusion coords:
    chipseq_exclusion_windows, exclusion_windows_bdt = utils.exclusion_regions(blacklist_file,
                                                                               chip_seq_coordinates)
    exclusion_windows_df = exclusion_windows_bdt.to_dataframe()
    # constructing the training set
    construct_sets = ConstructSets(genome_sizes_file=genome_sizes_file,
                                   genome_fasta_file=genome_fasta_file,
                                   blacklist_file=blacklist_file,
                                   chip_coords=chip_seq_coordinates,
                                   exclusion_df=exclusion_windows_df,
                                   window_length=window_lenght,
                                   curr_genome_bed=genome_bed_df,
                                   batch_size=batch_size,
                                   acc_regions_file=acc_regions_file,
                                   flanks=flanks,
                                   ratios=ratios)
    while True:
        X, y, coords = construct_sets.get_data()
        yield X, y


def get_test_data(genome_sizes_file, peaks_file, genome_fasta_file,
                  blacklist_file, to_keep, window_len, stride):

    ts = TestSet(genome_fasta_file=genome_fasta_file, genome_sizes_file=genome_sizes_file,
                 peaks_file=peaks_file, blacklist_file=blacklist_file,
                 window_len=window_len, stride=stride, to_keep=to_keep)
    X_test, y_test, coords = ts.get_data()
    return X_test, y_test, coords