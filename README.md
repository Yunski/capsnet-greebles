# cos429-capsnet
Comparing Capsule Networks with CNN architectures. \
Implementation of the capsule network is adapted from [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow).

# Getting Started
Install [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html).
Then create the environment with conda.
```python
# Use TensorFlow with GPU
conda env create -f environment-gpu.yml
```
### Download data
Change the dataset flag in config.py as needed.
Run the following script to download and extract data files.
```python
python data.py
```
Add a new entry to datasets.yml if you wish to test on another dataset not already included.

### Training and Testing
To train the network, set the is_training flag in config.py to True and run the script: 
```python
python capsule_experiment.py
```
To test the network, set the is_training flag in config.py to False. 
