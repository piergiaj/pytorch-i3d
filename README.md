# I3D model pre-trained on Kinetics and finetuned on <Jester, HVU>

## Overview
This repository was forked from https://github.com/piergiaj/pytorch-i3d, which provides a PyTorch I3D model trained on Kinetics. The pre-trained weights are used to finetune on 1) a modified version of the 20BN-jester dataset and 2) the Holistic Video Understanding dataset.

## Requirements
* pytorch
* torchvision
* tensorboard
* future
* numpy
* pandas
* ffmpeg
* PyAV --> https://github.com/mikeboers/PyAV#installation
