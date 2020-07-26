"""
Utilities for iterating constructing data sets and iterating over
DNA sequence data.
Some ideas and pybedtools code from:
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

from pybedtools import Interval, BedTool


class AccessGenome:
    def __init__(self, genome_sizes_file, genome_fasta_file):
        self.genome_sizes_file = genome_sizes_file
        self.genome_fasta_file = genome_fasta_file

    def get_genome_sizes(self):
        """
        Loads the genome sizes file which should look like this:
        chr1    45900011
        chr2    10001401
        ...
        chrX    9981013

        This function parses this file, and saves the resulting intervals file
        as a BedTools object.
        "Random" contigs, chrUns and chrMs are filtered out.

        Parameters:
            genome_sizes_file (str): (Is in an input to the class,
            can be downloaded from UCSC genome browser)
        Returns:
            A BedTools (from pybedtools) object containing all the chromosomes,
            start (0) and stop (chromosome size) positions
        """

        genome_sizes = pd.read_csv(self.genome_sizes_file, sep='\t',
                                   header=None, names=['chr', 'length'])
        # filter out chromosomes with "random" contigs
        genome_sizes_filt = genome_sizes[~genome_sizes['chr'].str.contains('random')]
        # filter out chromosomes with "chrUn" and "chrM"
        genome_sizes_filt = genome_sizes_filt[~genome_sizes_filt['chr'].str.contains('chrUn')]
        genome_sizes_filt = genome_sizes_filt[~genome_sizes_filt['chr'].str.contains('chrM')]

        genome_bed_data = []
        for chrom, sizes in genome_sizes_filt.values:
            genome_bed_data.append(Interval(chrom, 0, sizes))
        genome_bed_data = BedTool(genome_bed_data)
        return genome_bed_data

    def get_genome_fasta(self):
        f = pyfasta.Fasta(self.genome_fasta_file)
        return f


def load_chipseq_data(chip_peaks_file):
    """
    Loads the ChIP-seq peaks data.
    The chip peaks file is a tab seperated bed file:
    chr1    1   150
    chr2    2   350
    ...
    chrX    87  878
    This file can be constructed using a any peak-caller. We use multiGPS.
    Also constructs a BedTools object which can be later used to generate
    negative sets.

    """
    chip_seq_data = pd.read_csv(chip_peaks_file, sep='\t',
                                header=None,
                                names=['chr', 'start', 'end', 'caller',
                                       'score'])
    # also constructing a BedTools object, to intersect with negative data.
    chip_seq_bedtools_obj = BedTool(chip_peaks_file)
    return chip_seq_data, chip_seq_bedtools_obj


def exclusion_regions(blacklist_file, chip_seq_data):
    """
    This function takes as input a bound bed file (from multiGPS).
    The assumption is that the bed file reports the peak center
    For example: chr2   45  46
    It converts these peak centers into 501 base pair windows, and adds them to
    the exclusion list which will be used when constructing negative sets.
    It also adds the mm10 blacklisted windows to the exclusion list.

    Parameters:
        blacklist_file (str): Path to the blacklist file.
        chip_seq_data (dataFrame): The pandas chip-seq data loaded by load_chipseq_data
    Returns:
         exclusion_windows (BedTool): A bedtools object containing exclusion windows.
    """
    chip_seq_data['start'] = chip_seq_data['start'] - 250
    chip_seq_data['end'] = chip_seq_data['end'] + 250
    bound_exclusion_windows = BedTool.from_dataframe(chip_seq_data[['chr', 'start','end']])
    blacklist_exclusion_windows = BedTool(blacklist_file)
    exclusion_windows = BedTool.cat(
        *[blacklist_exclusion_windows, bound_exclusion_windows])
    return exclusion_windows


class ConstructTrainingSets(AccessGenome):

    def __init__(self, genome_sizes_file, genome_fasta_file, blacklist_file,
                 chip_coords, window_length):
        super().__init__(genome_sizes_file, genome_fasta_file)
        self.blacklist_file = blacklist_file
        self.chip_coords = chip_coords
        self.L = window_length

    def apply_random_shift(self):
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
            coords(pandas dataFrame): The output bedfile with shifted coords
        """
        # defining the random shift
        low = int(-self.L/2 + 25)
        high = int(self.L/2 - 25)
        self.chip_coords['random_shift'] = np.random.randint(low=low, high=high,
                                                   size=len(self.chip_coords))
        self.chip_coords['shifted_start'] = self.chip_coords['start'] + self.chip_coords['random_shift'] - int(self.L/2)
        self.chip_coords['shifted_end'] = self.chip_coords['start'] + self.chip_coords['random_shift'] + int(self.L/2)
        return self.chip_coords

    def define_coordinates(self):
        """
        Use the chip-seq peak file and the blacklist files to define a bound
        set and an unbound set of sites. The ratio of bound to unbound is 1:2,
        but can be controlled using the parameter "ratio".

        The unbound/negative set is chosen randomly from the genome.
        """
        # CHANGE THIS SO THAT YOU CAN PASS GENOME BED COORDS
        # INSTEAD OF CALLING IT OVER AND OVER
        chip_seq_data, chip_bedtools = self.load_chipseq_data()
        genome_bed_coords = super(ConstructTrainingSets, self).get_genome_sizes()
        batch_size = 100
        # note: genome_bed_coords is an instance of the pybedtools BedTool class

        # constructing a positive mini-batch from chip peaks file
        positive_set_size = int(batch_size/2)
        bound_indices = np.random.randint(low=0, high=len(chip_seq_data), size=positive_set_size)
        bound_bed = chip_seq_data.iloc[bound_indices]
        bound_bed = bound_bed.iloc[:, 0:3]  # extracting first 3 bed columns.
        shifted_coords = self.shift_windows(bound_bed)
        shifted_coords = shifted_coords[['chr', 'shifted_start', 'shifted_end']]

        # very very cool pybedtools functionality here!
        positive_bedtools_obj = BedTool.from_dataframe(shifted_coords)
        # print(positive_bedtools_obj)
        shifted_coords['label'] = 1


        # Note: with the include here make sure you include
        # the training chromosomes only.
        negative_bedtools_obj = BedTool.shuffle(positive_bedtools_obj,
                                       g=self.genome_sizes_file,
                                       excl=exclusion_windows.fn)
        negative_coords = negative_bedtools_obj.to_dataframe()
        negative_coords['label'] = 0
        negative_coords.columns = ['chr', 'shifted_start', 'shifted_end', 'label']


        # Mixing and shuffling positive and negative set:
        training_coords = pd.concat([shifted_coords, negative_coords])
        print(training_coords)
        # randomly shuffle the dataframe
        print(training_coords.sample(frac=1))



        # generating a random negative set:

        

        # removing any bound windows from the negative set:


        # joining the 2 sets.


        # shuffling and returning:

    def get_data_at_coordinates(self):
        # genome_fasta = super(ConstructSets, self).get_genome_fasta()
        # print(genome_fasta['chr1'][1:10])
        pass


mm10_sizes = '/Users/asheesh/Desktop/RNFs/mm10.sizes'
mm10_fa = '/Users/asheesh/Desktop/RNFs/mm10.fa'
peaks = '/Users/asheesh/Desktop/RNFs/Ascl1_Ascl1.bed'
mm10_blacklist = '/Users/asheesh/Desktop/RNFs/mm10_blacklist.bed'

construct_sets = ConstructTrainingSets(genome_sizes_file=mm10_sizes,
                               genome_fasta_file=mm10_fa,
                               blacklist_file=mm10_blacklist,
                               chip_peaks_file=peaks, window_length=200)

construct_sets.define_coordinates()










