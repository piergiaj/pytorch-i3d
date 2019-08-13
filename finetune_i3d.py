import os
import pdb
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from videotransforms import *
from dataset_torchvision import *
from pytorch_i3d import InceptionI3d
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets.ucf101 import UCF101


def train(model, optimizer, train_loader, test_loader, num_classes, max_steps, save_model='', use_gpu=False):
    # Enable GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using device:', device)
    model = model.to(device=device) # move model parameters to CPU/GPU

    dataloaders = {'train': train_loader, 'val': test_loader} 

    # Training loop
    num_steps_per_update = 4 # back-propagate every 4 steps
    steps = 0
    while steps < max_steps: # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  # set model to eval mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data
            for data in dataloaders[phase]:
                num_iter += 1
                inputs, _, class_idx = data # 2nd element of data tuple is audio
                inputs = inputs.permute(0, 4, 1, 2, 3) # swap from BxTxHxWxC to BxCxTxHxW
                inputs = inputs.to(device=device, dtype=torch.float32) # model expects inputs of float32

                # Forward pass
                per_frame_logits = model(inputs)
                print('per_frame_logits shape = {}'.format(per_frame_logits.shape))

                # Due to the strides and max-pooling in I3D, it temporally downsamples the video by a factor of 8
                per_frame_logits = F.interpolate(per_frame_logits, size=inputs.shape[2], mode='linear') # upsample to get per-frame predictions
                # Alternative: Take the average to get per-clip prediction
                
                # pdb.set_trace()

                # Convert ground-truth tensor to one-hot format
                labels = torch.zeros(per_frame_logits.shape)
                labels[np.arange(len(labels)), class_idx, :] = 1 # fancy broadcasting trick: https://stackoverflow.com/questions/23435782
                labels = labels.to(device=device)

                # Compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss

                # Compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss

                # Compute total loss and back-propagate
                loss = (0.5*loc_loss + 0.5*cls_loss) / num_steps_per_update
                tot_loss += loss
                loss.backward() 

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()

                    if steps % 10 == 0:
                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        torch.save(model.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.

            if phase == 'val':
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))


if __name__ == '__main__':
    # Parameters
    USE_GPU = True
    NUM_CLASSES = 101 # number of classes in UCF101
    FRAMES_PER_CLIP = 25 # UCF101 has a frame rate of 25 fps with a min clip length of 1.06 s
    STEPS_BETWEEN_CLIPS = 25
    FOLD = 1
    BATCH_SIZE = 8

    # Load dataset
    root = os.path.join(os.getcwd(), 'data/ucf101/clips')
    annotation_path = os.path.join(os.getcwd(), 'data/ucf101/ucfTrainTestlist')

    train_transform = T.Compose([videotransforms.RandomCrop(224)]) # I3D model expects input HxW of 224x224
    test_transform = T.Compose([videotransforms.CenterCrop(224)])
    
    d_train = UCF101(root,
                     annotation_path,
                     frames_per_clip=FRAMES_PER_CLIP,
                     step_between_clips=STEPS_BETWEEN_CLIPS,
                     fold=FOLD,
                     train=True,
                     transform=train_transform)
    train_loader = DataLoader(d_train, 
                              batch_size=BATCH_SIZE,
                              shuffle=True, 
                              num_workers=8,
                              pin_memory=True)

    d_test = UCF101(root,
                    annotation_path,
                    frames_per_clip=FRAMES_PER_CLIP,
                    step_between_clips=STEPS_BETWEEN_CLIPS,
                    fold=FOLD,
                    train=False,
                    transform=test_transform)
    test_loader = DataLoader(d_test,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=8,
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
    train(i3d, optimizer, train_loader, test_loader, 
          max_steps=64e3, num_classes=NUM_CLASSES, use_gpu=USE_GPU)
