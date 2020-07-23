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


class Construct:
    def __init__(self, genome_sizes_file, genome_fasta_file, blacklist_file,
                 chip_peaks_file):
        self.genome_sizes_file = genome_sizes_file
        self.genome_fasta_file = genome_fasta_file
        self.blacklist_file = blacklist_file
        self.chip_peaks_file = chip_peaks_file

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
        print(genome_bed_data)
        return genome_bed_data

    def get_genome_fasta(self):
        f = pyfasta.Fasta(self.genome_fasta_file)
        return f

    def define_coordinates(self, ratio=2):
        """
        Use the chip-seq peak file and the blacklist files to define a bound
        set and an unbound set of sites. The ratio of bound to unbound is 1:2,
        but can be controlled using the parameter "ratio".

        The unbound/negative set is chosen randomly from the genome.
        :return:
        """
        pass

    def get_data_at_coordinates(self):
        pass

## Pseudo code:
## Run: define coordinates
## get_data_at coordinates.
## Wrap this in an iterator.


construct_data = Construct('/Users/asheesh/Desktop/mm10.sizes', 1, 2, 3)
construct_data.get_genome_sizes()







