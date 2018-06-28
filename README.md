# I3D models trained on Kinetics

## Overview

This repository contains trained models reported in the paper "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman.

This code is based on Deepmind's [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d). Including PyTorch versions of their models.

## Note
This code was written for PyTorch 0.3. Version 0.4 and newer may cause issues.


# Fine-tuning and Feature Extraction
We provide code to extract I3D features and fine-tune I3D for charades. Our fine-tuned models on charades are also available in the models director (in addition to Deepmind's trained models). The deepmind pre-trained models were converted to PyTorch and give identical results (flow_imagenet.pt and rgb_imagenet.pt). These models were pretrained on imagenet and kinetics (see [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d) for details). 

## Fine-tuning I3D
[train_i3d.py](train_i3d.py) contains the code to fine-tune I3D based on the details in the paper and obtained from the authors. Specifically, this version follows the settings to fine-tune on the [Charades](allenai.org/plato/charades/) dataset based on the author's implementation that won the Charades 2017 challenge. Our fine-tuned RGB and Flow I3D models are available in the model directory (rgb_charades.pt and flow_charades.pt).

This relied on having the optical flow and RGB frames extracted and saved as images on dist. [charades_dataset.py](charades_dataset.py) contains our code to load video segments for training.

## Feature Extraction
[extract_features.py](extract_features.py) contains the code to load a pre-trained I3D model and extract the features and save the features as numpy arrays. The [charades_dataset_full.py](charades_dataset_full.py) script loads an entire video to extract per-segment features.
