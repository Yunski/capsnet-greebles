# cos429-capsnet
Comparing Traditional CNNs and Capsule Networks with Greebles Classification. \
Implementation of the capsule network is adapted from [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow).

# Getting Started
Install [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html).
Then create the environment with conda.
```
# Use TensorFlow with GPU
$ conda env create -f environment-gpu.yml
```
### Download data
Change the dataset flag in config.py as needed.
Run the following script to download and extract data files.
```
$ python data.py
```
For affNIST and smallnorb, also run the scripts:
```
$ python affnist.py
$ python smallnorb.py
```
For Greebles, generate a greebles dataset with the script found at [sherrybai/greebles-generator](https://github.com/sherrybai/greebles-generator). 
Extract the resulting zip file into the directory `data/greebles`.
Then, run the script:
```
$ python greebles.py
```

Add a new entry to datasets.yml if you wish to test on another dataset not already included.

### Training and Testing
Training a model: 
```
$ python {model}_train.py 
```
Testing a model: 
```
$ python {model}_eval.py
```
i.e. `python cnn_train.py`

Logs and train/validation/test csv files can be found under logs/ and summary/. \
Visualize plots with tensorboard:
```
$ tensorboard --logdir=logdir
```
