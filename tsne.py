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

# ----------------- Modify these --------------------

NUM_ACTIONS = 5
NUM_SCENES = 2
NUM_FEATURES = 1024
BATCH_SIZE = 128

""" baseline i3d params """
IS_BASELINE = True # use baseline i3d
DATA_PARALLEL = True # model trained using nn.DataParallel
CHECKPOINT_PATH = '/vision/u/rhsieh91/pytorch-i3d/checkpoints-2019-12-9-22-36-12/22085238.pt' # epoch 22
FEATURES_PATH = '/vision/u/samkwong/pytorch-i3d/input_features_i3d_epoch22.npy'
FEATURES_SAVE_PATH = 'input_features_i3d_epoch22' # features will only be saved if FEATURES_PATH is defined
ACTIONS_PATH = '/vision/u/samkwong/pytorch-i3d/input_actions_i3d_epoch22.npy'
ACTIONS_SAVE_PATH = 'input_actions_i3d_epoch22'
TSNE_ACTION_SAVE_PATH = 'tsne_i3d_action_jester_epoch22.png'

""" sife params """
#IS_BASELINE = False # use sife
#DATA_PARALLEL = False # model trained without using nn.DataParallel
#CHECKPOINT_PATH = '/vision/u/samkwong/pytorch-i3d/checkpoints-2019-12-9-15-47-16/22170453.pt' # epoch 22
#FEATURES_PATH = '/vision/u/samkwong/pytorch-i3d/input_features_sife_epoch22.npy'
#FEATURES_SAVE_PATH = 'input_features_sife_epoch22' # features will only be saved if FEATURES_PATH is defined
#ACTIONS_PATH = '/vision/u/samkwong/pytorch-i3d/input_actions_sife_epoch22.npy'
#ACTIONS_SAVE_PATH = 'input_actions_sife_epoch22'
#SCENES_PATH = '/vision/u/samkwong/pytorch-i3d/input_scenes_sife_epoch22.npy'
#SCENES_SAVE_PATH = 'input_scenes_sife_epoch22'
#TSNE_ACTION_SAVE_PATH = 'tsne_sife_action_jester_epoch22.png'
#TSNE_SCENE_SAVE_PATH = 'tsne_sife_scene_jester_epoch22.png'


def load_checkpoint():
    print("Loading checkpoint...")
    if DATA_PARALLEL: # If training was run with nn.DataParallel, need extra steps before loading checkpoint
        print("Used nn.DataParallel")
        state_dict = torch.load(CHECKPOINT_PATH)['model_state_dict'] # baseline weights
        checkpoint = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module'
            checkpoint[name] = v
    else:
        print('Using sife weights')
        checkpoint = torch.load(CHECKPOINT_PATH)['model_state_dict'] # sife weights
    model.load_state_dict(checkpoint)
    print("Loaded")

def extract_data(model, test_loader):
    # Move model to CPU/GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Using device:', device)
    model = model.to(device=device) # move model parameters to CPU/GPU

    # Extract features and ground truth labels
    print('Starting feature extraction with batch size = {}'.format(BATCH_SIZE))
    inputs_features = np.empty((0, NUM_FEATURES)) # to hold all inputs' feature arrays
    inputs_actions = np.empty(0)
    inputs_scenes = np.empty(0) # only used if not using baseline
    for i, data in enumerate(test_loader):
        print("Extracting features from batch {}".format(i))
        inputs, action_idxs = data[0], data[1]
        if not IS_BASELINE:
            scene_idxs = data[2]
        inputs = inputs.to(device=device, dtype=torch.float32) 
        with torch.no_grad():
            if IS_BASELINE:
                features = model.extract_features(inputs)
            else:
                features = model.backbone.extract_features(inputs) # using SIFE with I3D backbone

        features = features.squeeze()
        features = features.cpu().detach().numpy()
        action_idxs = action_idxs.squeeze().cpu().detach().numpy()
        if not IS_BASELINE:
            scene_idxs = scene_idxs.squeeze().cpu().detach().numpy()
        inputs_features = np.append(inputs_features, features, axis=0)
        inputs_actions = np.append(inputs_actions, action_idxs, axis=0)
        if not IS_BASELINE:
            inputs_scenes = np.append(inputs_scenes, scene_idxs, axis=0)

    print('inputs_features shape = {}'.format(inputs_features.shape))
    print('Saving features')
    np.save(FEATURES_SAVE_PATH, inputs_features)
    np.save(ACTIONS_SAVE_PATH, inputs_actions)
    if not IS_BASELINE:
        np.save(SCENES_SAVE_PATH, inputs_scenes)
    return inputs_features, inputs_actions, inputs_scenes

def get_test_loader(model):
    # Transforms
    SPATIAL_TRANSFORM = Compose([
        Resize((224, 224)),
        ToTensor()
        ])
    
    # Load dataset
    vf = VideoFolder(root="/vision/group/video/scratch/jester/rgb",
                          csv_file_input="./data/jester/annotations/jester-v1-train-modified.csv",
                          csv_file_action_labels="./data/jester/annotations/jester-v1-action-labels.csv",
                          csv_file_scene_labels="./data/jester/annotations/jester-v1-scene-labels.csv",
                          clip_size=16,
                          nclips=1,
                          step_size=1,
                          is_val=True, # True means don't randomly offset clips (i.e. don't augment dataset)
                          transform=SPATIAL_TRANSFORM,
                          loader=default_loader)

    print('Size of training set = {}'.format(len(vf)))
    test_loader = DataLoader(vf, 
                              batch_size=BATCH_SIZE,
                              shuffle=False, 
                              num_workers=2,
                              pin_memory=True)

    return test_loader

def plot_tsne(inputs_truths, colors, labels, save_path):
    for i, c, label in zip(range(len(colors))[::-1], colors[::-1], labels[::-1]):
        fig = plt.scatter(features_embedded[inputs_truths == i, 0], features_embedded[inputs_truths == i, 1], c=c, label=label) 
    plt.legend()
    plt.savefig(save_path)
    fig.remove()

# ------------------------------------------------------------

# Either load features from disk or compute them
if FEATURES_PATH:
    inputs_features = np.load(FEATURES_PATH)
    inputs_actions = np.load(ACTIONS_PATH)
    if not IS_BASELINE:
        inputs_scenes = np.load(SCENES_PATH)
    print('Loaded saved features and ground truth labels')
else:
    i3d = InceptionI3d(400, in_channels=3)
    if IS_BASELINE:
        model = i3d
        model.replace_logits(NUM_ACTIONS)
    else:
        model = SIFE(backbone=i3d, num_features=NUM_FEATURES, num_actions=NUM_ACTIONS, num_scenes=NUM_SCENES)

    load_checkpoint()
    test_loader = get_test_loader(model)
    inputs_features, inputs_actions, inputs_scenes = extract_data(model, test_loader)

# Calculate TSNE
print("Starting TSNE")
features_embedded = TSNE(n_jobs=8).fit_transform(inputs_features) # MultiCoreTSNE automatically uses n_components=2
print("Finished TSNE")
print('feautures_embedded shape = {}'.format(features_embedded.shape))

# Plot TSNE for action
print("Plotting action TSNE")
action_colors = ['r', 'g', 'b', 'c', 'grey'] # create color list with num elements equal to num action labels 
action_labels = ['swiping-left', 'swiping-right', 'swiping-down', 'swiping-up', 'other']
plot_tsne(inputs_actions, action_colors, action_labels, TSNE_ACTION_SAVE_PATH)

if not IS_BASELINE:
    # Plot TSNE for scene
    print("Plotting scene TSNE")
    scene_colors = ['orange', 'grey'] # create color list with num elements equal to num scene labels
    scene_labels = ['swiping', 'other']
    plot_tsne(inputs_scenes, scene_colors, scene_labels, TSNE_SCENE_SAVE_PATH)
    
