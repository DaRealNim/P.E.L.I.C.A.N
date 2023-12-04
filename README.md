# PELICAN
*(**P**redictive **E**ngine for **L**earning and **I**dentification of **C**yber **A**nomalies and **N**uisances)*

## Overview
PELICAN is a machine learning model for binary malware classification of windows Portable Executables. It is inspired by the architecture described in an [NVIDIA blog post on AI malware detection](https://developer.nvidia.com/blog/malware-detection-neural-networks/), but implements some regularization technices, and a slightly different architecture.

The architecture is as follows:
- 8 dimensional embeddings are learned for each byte
- A gated convolutional layer with 128 filters, a kernel size and stride of 500
- Global max pooling
- A fully connected layer

## Current best results
For the given task, PELICAN achieves the following metrics on the test set (20% of total dataset, "malware" is the positive class) after 8 epochs of training:

| Accuracy | Precision | Recall | F1   | ROC AUC |
|----------|-----------|--------|------|---------|
| 0.96     | 0.92      | 0.93   | 0.92 | 0.98    |

## Setup
Install the dependencies in `requirements.txt` using `pip install -r requirements.txt`. 

## Usage 
To use the model:

```bash
python use.py -m <path to model> -i <path to input file or directory> [-r]
```
The `-i` argument accepts wildcards, but must in that case be quoted. The `-r` argument is optional and will recursively go through directories.

## Training

To train the model:

```bash
python train.py [-e EPOCHS] [-b BATCH_SIZE] [-a ACCUMULATE] [-l LEARNING_RATE] [-d DATAPKL] [-c CHECKPOINT_PATH] [-o OUTPUT_PATH] [--random-seed RANDOM_SEED] [--test-ratio TEST_RATIO] [--device DEVICE]
```

Options:
- `-e`, `--epochs`: Specify the number of epochs for training. Default is 10.
- `-b`, `--batch_size`: Set the batch size. Default is 64.
- `-a`, `--accumulate`: Define how often to accumulate gradients per batch. Default is 1.
- `-l`, `--learning_rate`: Set the learning rate. Default is 0.01.
- `-d`, `--datapkl`: Path to the data pickle file. Default is 'data.pkl'.
- `-c`, `--checkpoint-path`: Specify the checkpoint path. Default is 'checkpoint'.
- `-o`, `--output-path`: Set the model output path. Default is 'output'.
- `--random-seed`: Set the random seed for data splitting and model initialization. Default is 42.
- `--test-ratio`: Specify the ratio of test data. Default is 0.2.
- `--device`: Choose the device to use (cuda or cpu). Default is 'cuda'.

## Weights
The `weights` directory contains the parameters of the model at the epoch with the best test metrics during training. The model was trained on a P100 GPU on Kaggle.

## Data
The data was not included in the repository due to its size. The data input of the `train.py` script is a pickle file containing a pandas dataframe with the following columns:
- `label`: the label of the file, "malware" or "goodware"
- `text_bytes`: the bytes of the `.text` section of the PE file, as a python bytes object

## Notebooks
The `notebooks` directory contains Jupyter notebooks for experiments for the project.
- `experiments.ipynb` contains data analysis, discovery of a bias probably linked to file entropy and length, and results with simple models (including samples from The-Malware-Repo).
- `bytetokenizer.ipynb` contains an attempt to recreate byte-pair encoding to make it work on arbitrary byte sequences, as the HuggingFace tokenizer only works on text. The goal was to try to find meaningful tokens to then process with an LSTM or Transformer. Infortunately my implementation is too slow to be usable with the large sequences of bytes in the dataset, as I didn't implement efficient updates of the pair counts during the application of merge rules, which meant I had to recompute the counts from scratch at each step.
- `kaggle_train.ipynb` contains the latest training of the model, that was run on Kaggle to take advantage of the GPU.
- `lstmmodel.ipynb` contains code for an aborted attempt at using an LSTM to interpret byte sequences. Unfortunately, without a proper tokenizer, the sequences were too long, the model too slow to train, and the results were not good. The training loop was a bit different, with loss function masking to ignore padding. An attempt at using a Transformer was also made, but lack of time and computational resources prevented me from training it properly.