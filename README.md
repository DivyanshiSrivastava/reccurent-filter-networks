# Reccurrent Filter Networks
DNA sequences contain short sequence motifs that bind transciption factors (TF). Convolutional neural networks (CNNs) have been widely used to learn such sequence motifs in DNA to predict TF binding. However, the convolution operation itself may not be able to capture inter-nucleotide dependencies in sequence motifs, which in turn may drive TF binding specificity. Here, we employ 1st order and 2nd order RNNs to model the sequence motif predictors of TF binding. 

## Installation
**Requirements**:  

python >= 3.5  
We suggest using anaconda to create a virtual environment using the provided YAML configuration file:
`conda env create -f recurrent_env.yml`  
Alternatively, to install requirements using pip: 
`pip install -r requirements.txt`

## Usage
```
# Clone and navigate to the iTF repository. 
cd recurrent-filter-networks  
To view help:   
python train.py --help
usage: train.py [-h] [--params_yaml]
                [--acc_regions_file]
                [--network_type]
                genome_sizes blacklist fa peaks results_dir
```

**Input Data**
**Required Inputs:**
* *genome_size*: A standard file that records the chromosome sizes for each chromosome in the genome of interest. 
* *blacklist*: A standard BED format blacklist file. Blacklist files for the human hg38 and hg37 genomes can be found here: https://sites.google.com/site/anshulkundaje/projects/blacklists. Blacklist files for other commonly used genome annotations are here: https://github.com/Boyle-Lab/Blacklist/tree/master/lists
* *fa*: The complete fasta file for the genome of interest. 
* *peaks*: File containing ChIP-seq peaks in the multiGPS events format. Each row records one ChIP-seq peak. 
  Sample peaks file: 
  ```
  chr1:247890
  chr7:1288919
  ...
  chrX:89129
  ```
* *results_dir*: Output directory for storing the Model performance. 

**Optional Inputs:** 
* --params_yaml: YAML file that contains network hyper-parameters. If this argument is not provided, default parameter settings will be used. 
Sample YAML file: 
```
parameters:
  batch_size: 512
  ratio: [5, 6, 4]
  patience: 8
  lr: 0.0008
  optimizer: 'adam'
  steps: 2500
  filter_width: 20
  no_of_filters: 128
```
* --acc_regions_file: A BED file with accessible genomics regions. If this argument is provided, the selection procedure for negative training batches does not consider chromatin accessibility.

* --network_type: 'RNF' or 'CNN'. If this argument is not provided, a default recurrent network is trained. 

