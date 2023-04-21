# Training code for Kaggle - IceCube - Neutrinos in Deep Ice

This repository contains training code for my 3rd place at Kaggle Competition - IceCube - Neutrinos in Deep Ice.

I may add some improvements and new ideas in the future, from other participant solutions that I found insightful.

## Links

- [Competition Page](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/)
- [Solution Writeup](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/discussion/402888)

## Overview

The goal of the Icecube Challenge is to predict a neutrino particle’s direction, with sensor data from the Icecube neutrino detector located in Antarctica.

Problem statement: Given a sequence of detections containing information about the coordinates of detection, time of detection, and intensity of the light, predict two angles (Azimuth and Zenith) of the incoming neutrino that caused the event.

My final solution used for the challenge used the following components

- Self Attention Encoder
- Two angle classification heads
- 3D vector regression head - trained with Von Mises-Fisher Loss
- Gradient Boosting Based ensembler to combine the predictions

## Data Preparation

### Download the data

Download the data from the [competition page](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/data)

You can place the data anywhere you like, by default the training code looks for the the environment variable `ICECUBE_DATA_DIR` or else checks in `./data`

### Preprocessing

Run the data preparation scripts in the `data_preparation` directory

- `angle_bins.py`
- `geometry_with_transparency.py`
- `split_meta.py`

## Training

### Base model training

Hyperparameters in `attention_model_training/config.py` are for the final model which was trained for 5 days on a RTX 4080.

Original Model Parameters:
  
- Embedding size - 512
- Heads - 8
- Layers - 18 (Longer network proved better for Icecube)
- Average Pooling
- Neck Size - 3072
- 128 bins for angle classification

Other details:

- Loss - Locally Smoothed Cross Entropy
- LR Schedule - OneCycle (2 phase cosine)
- Max LR - 5e-5
- FP16 training (BF16 was unstable)
- Switch to FP32 towards end of training
- Effective batch size - 384 (32 x 12)

You can reduce the model size to get a model that reaches score ~1.00 in 3 hours on RTX 4080

Small model details

- Embedding size - 64
- Heads 2
- Layers - 9
- Neck Size - 1024
- Max LR - 2e-4
- Batch size 384 (Accumulate 1)

Model training does **length based grouping**, this gives a training speed gain.

For very long sequences, fine tune the trained model on lengths upto 3072 (see FINETUNE_LONG_SEQ in the config file)

### Stack Model training

For efficiency I dump 100 batches of encoder predictions before training the stack model. The stack model is a simple MLP trained to do 3D vector regression with Von Mises-Fisher Loss.

Run inference with `inference/inference.py`

Set the cached prediction paths in `stack_model_training/config.py`

Train stack model with `stack_model_training/train.py`

## Extra notes

- Using Flash Attention - This was still early stages when I started writing the code. For some reason default pytorch flash attention was not working even though the original flash attention repo worked. If you're having trouble installing flash attention, you can revert to default pytorch attention by making necessary changes in `attention_model_training/attn_model.py`
- Hyperparameter sensitivity:
  - LR Schedule - Fairly sensitive to changining the LR
  - Classifier bins - Going above 128 bins causes some instability
  - Dropout - Always results in worse performance
  - Weight decay - Can change without too much effect
  - Network size
    - Embedding/Head - Below 32 didn't work well
    - Layers - 9 layers gave big improvement over 6

- Mixed precision instability - I saw some instability in training after validation metric reached 0.992, this has not been further investigated due to lack of time. I just switch to FP32 (code supports automatic switching after a set epoch number)

## Acknowledgements

- The attention model code was heavily adopted from [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT)
- Lots of insights about data processing and loss functions was taken from [GraphNet](https://github.com/graphnet-team/graphnet)
- Kaggle discussion posts and baselines were extremely helpful.
