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

# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from collections import OrderedDict


# Modify these
NUM_ACTIONS = 5
NUM_SCENES = 2
NUM_FEATURES = 1024
BATCH_SIZE = 128
ADVERSARIAL = False
DATA_PARALLEL = True # Was model trained using nn.DataParallel?
CHECKPOINT_PATH = '/vision/u/rhsieh91/pytorch-i3d/checkpoints-2019-12-9-22-36-12/04018530.pt'
FEATURES_PATH = '/vision/u/rhsieh91/pytorch-i3d/inputs_features_baseline.npy'
FEATURES_SAVE_PATH = 'input_features_baseline' # features will only be saved if FEATURES_PATH is defined
TSNE_SAVE_PATH = 'tsne_baseline.png'

# Don't need to modify these
NUM_WORKERS = 2
SHUFFLE = False
PIN_MEMORY = True
IS_VAL = True # True = don't randomly offset clips (i.e. don't augment dataset)

# Either load features from disk or compute them
if FEATURES_PATH:
    inputs_features = np.load(FEATURES_PATH)

else:
    i3d = InceptionI3d(400, in_channels=3)
    if not ADVERSARIAL:
        model = i3d
        model.replace_logits(NUM_ACTIONS)
    else:
        model = SIFE(backbone=i3d, num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, num_scenes=SCENES)

    # Load checkpoint
    print("Loading checkpoint...")
    if DATA_PARALLEL: # If training was run with nn.DataParallel, need extra steps before loading checkpoint
        state_dict = torch.load(CHECKPOINT_PATH)['model_state_dict'] # baseline weights
        checkpoint = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module'
            checkpoint[name] = v
    else:
         checkpoint = torch.load(CHECKPOINT_PATH)['model_state_dict'] # adversarial weights
    model.load_state_dict(checkpoint)
    print("Loaded.")

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
                          is_val=IS_VAL,
                          transform=SPATIAL_TRANSFORM,
                          loader=default_loader)

    print('Size of training set = {}'.format(len(d_train)))
    train_loader = DataLoader(d_train, 
                              batch_size=BATCH_SIZE,
                              shuffle=SHUFFLE, 
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)

    # Move model to CPU/GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Using device:', device)
    model = model.to(device=device) # move model parameters to CPU/GPU

    # Either load features from disk or compute them
    if FEATURES_PATH:# Load inputs_features from disk
        inputs_features = np.load(FEATURES_PATH)
    else:
        print('Starting feature extraction with batch size = {}'.format(BATCH_SIZE))

        inputs_features = np.empty((0, NUM_FEATURES)) # to hold all inputs' feature arrays
        for i,  data in enumerate(train_loader):
            print("Extracting features from batch {}".format(i))
            inputs = data[0]
            inputs = inputs.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                if not ADVERSARIAL:
                    features = model.extract_features(inputs)
                else:
                    features = model.backbone.extract_features(inputs)

            features = features.squeeze()
            features = features.cpu().detach().numpy()
            inputs_features = np.append(inputs_features, features, axis=0) 

        print('inputs_features shape = {}'.format(inputs_features.shape))
        print('Saving features')
        np.save(FEATURES_SAVE_PATH, inputs_features)


# Calculate TSNE
print("Starting TSNE")
features_embedded = TSNE(n_jobs=8).fit_transform(inputs_features) # MultiCoreTSNE automatically uses n_components=2
print("Finished TSNE")
print('feautures_embedded shape = {}'.format(features_embedded.shape))

# Plot TSNE
colors = np.arange(features_embedded.shape[0]) # create color array with num elements equal to num samples in features_embeddedd
plt.scatter(features_embedded[:,0], features_embedded[:,1], marker='.', c=colors, cmap=plt.cm.get_cmap('viridis'))
plt.colorbar()
plt.savefig(TSNE_SAVE_PATH)
