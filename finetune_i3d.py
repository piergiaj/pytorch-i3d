import os
import pdb
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pytorch_i3d import InceptionI3d
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from ucf101 import UCF101
from spatial_transforms import Compose, ToTensor, Scale


def train(model, optimizer, train_loader, test_loader, num_classes, epochs, save_dir='', use_gpu=False):
    # Enable GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using device:', device)
    model = model.to(device=device) # move model parameters to CPU/GPU

    dataloaders = {'train': train_loader, 'val': test_loader} 

    # Training loop
    for e in range(epochs):    
        print('Epoch {}/{}'.format(e, epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  # set model to eval mode
            
            # Iterate over data
            for t, data in enumerate(dataloaders[phase]):
                print('Step {}:'.format(t))
                inputs = data[0] # input shape = B x C x T x H x W
                inputs = inputs.to(device=device, dtype=torch.float32) # model expects inputs of float32
                # print('inputs shape = {}'.format(inputs.shape))

                # Forward pass
                per_frame_logits = model(inputs) 
                # print('per_frame_logits shape = {}'.format(per_frame_logits.shape))

                # Due to the strides and max-pooling in I3D, it temporally downsamples the video by a factor of 8
                # so we need to upsample (F.interpolate) to get per-frame predictions
                # ALTERNATIVE: Take the average to get per-clip prediction
                per_frame_logits = F.interpolate(per_frame_logits, size=inputs.shape[2], mode='linear') # output shape = B x NUM_CLASSES x T

                # Convert ground-truth tensor to one-hot format
                class_idx = data[1]['label']
                labels = torch.zeros(per_frame_logits.shape)
                labels[np.arange(len(labels)), class_idx, :] = 1 # fancy broadcasting trick: https://stackoverflow.com/questions/23435782
                labels = labels.to(device=device)
                # print('labels shape = {}'.format(labels.shape))

                # Compute classification loss (max along time T)
                loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])

                # Backward pass
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                if t % 20 == 0:
                    print('{}, Loss = {}'.format(phase,loss))

                    save_path = save_dir + str(t).zfill(6) + '.pt'
                    torch.save({
                                'epoch': e,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss
                                },
                                save_path)

            # TODO: Function to check accuracy on test set


def check_accuracy(model, test_loader):
    pass


if __name__ == '__main__':
    # Parameters
    USE_GPU = True
    NUM_CLASSES = 101 # number of classes in UCF101
    FOLD = 1
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    SHUFFLE = False
    SAVE_DIR = 'checkpoints/'

    # Transforms
    SPATIAL_TRANSFORM = Compose([
        Scale((224, 224)),
        ToTensor()
        ])

    # Load dataset
    video_path = '/vision/u/rhsieh91/UCF101/jpg'
    annotation_path = '/vision/u/rhsieh91/UCF101/ucfTrainTestlist/ucf101_0' + str(FOLD) + '.json'
    
    d_train = UCF101(video_path,
                     annotation_path,
                     subset='training',
                     n_samples_for_each_video=4,
                     spatial_transform=SPATIAL_TRANSFORM)
    train_loader = DataLoader(d_train, 
                              batch_size=BATCH_SIZE,
                              shuffle=SHUFFLE, 
                              num_workers=NUM_WORKERS,
                              pin_memory=True)

    d_val = UCF101(video_path,
                   annotation_path,
                   subset='validation',
                   n_samples_for_each_video=4,
                   spatial_transform=SPATIAL_TRANSFORM)
    val_loader = DataLoader(d_val, 
                            batch_size=BATCH_SIZE,
                            shuffle=SHUFFLE, 
                            num_workers=NUM_WORKERS,
                            pin_memory=True)
    
    # Load pre-trained I3D model
    i3d = InceptionI3d(400, in_channels=3) # pre-trained model has 400 output classes
    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(NUM_CLASSES) # replace final layer to work with new dataset

    # Set up optimizer
    optimizer = optim.Adam(i3d.parameters(), lr=0.01)
    # optimizer = optim.SGD(i3d.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0000001)
    # lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    # Start training
    train(i3d, optimizer, train_loader, val_loader, num_classes=NUM_CLASSES, epochs=5, save_dir=SAVE_DIR, use_gpu=USE_GPU)
