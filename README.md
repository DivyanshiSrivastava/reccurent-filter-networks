# Reccurrent Filter Networks
DNA sequences contain short sequence motifs that bind transciption factors (TF). Convolutional neural networks (CNNs) have been widely used to learn such sequence motifs in DNA to predict TF binding. However, the convolution operation itself may not be able to capture inter-nucleotide dependencies in sequence motifs, which in turn may drive TF binding specificity. Here, we employ 1st order and 2nd order RNNs to model the sequence motif predictors of TF binding. 

## Installation
**Requirements**:  

python >= 3.5  
We suggest using anaconda to create a virtual environment using the provided YAML configuration file:
`conda env create -f recurrent_env.yml`  
Alternatively, to install requirements using pip: 
`pip install -r requirements.txt`
