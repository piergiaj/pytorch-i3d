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
from torch.utils.tensorboard import SummaryWriter


def train(model, optimizer, train_loader, test_loader, num_classes, epochs, save_dir='', use_gpu=False):
    # Enable GPU if available
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    model = model.to(device=device) # move model parameters to CPU/GPU

    dataloaders = {'train': train_loader, 'val': test_loader} 

    writer = SummaryWriter() # Tensorboard logging

    # Training loop
    n_iter = 0
    for e in range(epochs):    
        print('Epoch {}/{}'.format(e, epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  # set model to eval mode
            
            num_correct = 0 # keep track of number of correct predictions
            best_val = -1 # keep track of best val accuracy seen so far

            # Iterate over data
            for data in dataloaders[phase]:
                print('{}, step {}:'.format(phase, n_iter))
                inputs = data[0] # input shape = B x C x T x H x W
                inputs = inputs.to(device=device, dtype=torch.float32) # model expects inputs of float32

                # Forward pass
                per_frame_logits = model(inputs) 

                # Due to the strides and max-pooling in I3D, it temporally downsamples the video by a factor of 8
                # so we need to upsample (F.interpolate) to get per-frame predictions
                # ALTERNATIVE: Take the average to get per-clip prediction
                per_frame_logits = F.interpolate(per_frame_logits, size=inputs.shape[2], mode='linear') # output shape = B x NUM_CLASSES x T

                # Convert ground-truth tensor to one-hot format
                class_idx = data[1]['label'] # shape = B
                class_idx = class_idx.to(device=device)
                labels = torch.zeros(per_frame_logits.shape)
                labels[np.arange(len(labels)), class_idx, :] = 1 # fancy broadcasting trick: https://stackoverflow.com/questions/23435782
                labels = labels.to(device=device)

                # Count number of correct predictions
                _, argmax = torch.max(per_frame_logits, dim=1) # argmax shape = B x T
                pred, _ = torch.max(argmax, dim=1) # pred shape = B x 1
                num_correct += torch.sum(pred == class_idx)

                # Backward pass only if in 'train' mode
                if phase == 'train':
                    # Compute classification loss (max along time T)
                    loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                    writer.add_scalar('Loss/train', loss, n_iter)
                    
                    optimizer.zero_grad()
                    loss.backward() 
                    optimizer.step()

                    if n_iter % 10 == 0:
                        print('{}, loss = {}'.format(phase, loss))
                    n_iter += 1
                
                if n_iter % 100 == 0:
                    save_path = save_dir + str(e).zfill(2) + str(n_iter).zfill(6) + '.pt'
                    torch.save({
                                'epoch': e,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss
                                },
                                save_path)

                if n_iter % 50 == 0:
                    break

            # Log train/val accuracy
            accuracy = num_correct / len(dataloaders[phase].dataset)
            print('num_correct = {}'.format(num_correct))
            print('{}, accuracy = {}'.format(phase, accuracy))
            if phase == 'train':
                writer.add_scalar('Accuracy/train', accuracy, e)
            else:
                writer.add_scalar('Accuracy/val', accuracy, e)

    writer.close()       


if __name__ == '__main__':
    # Parameters
    USE_GPU = True
    NUM_CLASSES = 101 # number of classes in UCF101
    FOLD = 1
    BATCH_SIZE = 16
    NUM_WORKERS = 1
    SHUFFLE = True
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
    print('Size of training set = {}'.format(len(d_train)))
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
    print('Size of validation set = {}'.format(len(d_val)))
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
    train(i3d, optimizer, train_loader, val_loader, num_classes=NUM_CLASSES, epochs=2, save_dir=SAVE_DIR, use_gpu=USE_GPU)
