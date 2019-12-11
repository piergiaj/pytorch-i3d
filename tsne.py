import os
import pdb
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pytorch_i3d import InceptionI3d
from pytorch_sife import SIFE
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.tensorboard import SummaryWriter

from data_loader_jpeg import *

from sklearn.manifold import TSNE


i3d = InceptionI3d(400, in_channels=3) # pre-trained model has 400 output classes
i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
sife = SIFE(backbone=i3d, num_features=1024, num_actions=5, num_scenes=2)

print("Loading checkpoint...")
checkpoint = torch.load('./checkpoints-2019-12-9-15-47-16/02022233.pt')['model_state_dict']
print("Loaded.")

sife.load_state_dict(checkpoint)
    
NUM_ACTIONS = 5
NUM_SCENES = 2
NUM_WORKERS = 2
BATCH_SIZE = 16
SHUFFLE = True
PIN_MEMORY = True

# Transforms
SPATIAL_TRANSFORM = Compose([
    Resize((224, 224)),
    ToTensor()
    ])

# Load dataset
d_train = VideoFolder(root="/vision/group/video/scratch/jester/rgb",
                      csv_file_input="./data/jester/annotations/jester-v1-train-modified.csv",
                      csv_file_action_labels="./data/jester/annotations/jester-v1-action-labels.csv",
                      csv_file_scene_labels="./data/jester/annotations/jester-v1-scene-labels.csv",
                      clip_size=16,
                      nclips=1,
                      step_size=1,
                      is_val=False,
                      transform=SPATIAL_TRANSFORM,
                      loader=default_loader)

print('Size of training set = {}'.format(len(d_train)))
train_loader = DataLoader(d_train, 
                          batch_size=BATCH_SIZE,
                          shuffle=SHUFFLE, 
                          num_workers=NUM_WORKERS,
                          pin_memory=PIN_MEMORY)

# TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print('Using device:', device)

# if torch.cuda.device_count() > 1:
#     print('Multiple GPUs detected: {}'.format(torch.cuda.device_count()))
#     sife = nn.DataParallel(sife)
# AttributeError: 'DataParallel' object has no attribute 'backbone'

sife = sife.to(device=device) # move model parameters to CPU/GPU

inputs_features = [] # to hold all inputs' feature arrays
i = 0
for data in train_loader:
    print("Extracting features from batch {}".format(i))
    inputs = data[0]
    inputs = inputs.to(device=device, dtype=torch.float32)
    # TODO (after extracting 2nd batch's features) RuntimeError: CUDA out of memory. Tried to allocate 98.00 MiB (GPU 0; 11.91 GiB total capacity; 11.23 GiB already allocated; 19.12 MiB free; 127.04 MiB cached)
    features = sife.backbone.extract_features(inputs)
    features = features.squeeze()
    inputs_features.append(features)
    i += 1

print("Starting TSNE")
features_embedded = TSNE(n_components=2).fit_transform(inputs_features.cpu().detach().numpy()) # might need >1024 features eventually (at least 8000)
print("Finished TSNE.")
# TODO plot features_embedded
