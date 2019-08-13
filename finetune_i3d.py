import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import videotransforms

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
# from dataset import TSNDataSet
from pytorch_i3d import InceptionI3d
from transforms import Stack, ToTorchFormatTensor

# newly released UCF101 dataset on pytorch=1.2.0 and torchvision=0.4
from torchvision.datasets.ucf101 import UCF101


USE_GPU = True
NUM_CLASSES = 101


def train(fold=1, init_lr=0.1, max_steps=64e3, save_model='', use_gpu=False):
    """
    Load pretrained I3D model and finetune on new dataset.
    """
    # Enable GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Load data
    root = os.path.join(os.getcwd(), 'data/ucf101/clips')
    annotation_path = os.path.join(os.getcwd(), 'data/ucf101/ucfTrainTestlist')

    training_set = UCF101(root=root,
                          annotation_path=annotation_path,
                          frames_per_clip=16,
                          step_between_clips=16,
                          fold=fold,
                          train=True,
                          transform=T.Compose([videotransforms.RandomCrop(224)]))
    test_set = UCF101(root=root,
                      annotation_path=annotation_path,
                      frames_per_clip=16,
                      step_between_clips=16,
                      fold=fold,
                      train=False,
                      transform=T.Compose([videotransforms.RandomCrop(224)]))

    loader_train = DataLoader(training_set)
    loader_test = DataLoader(test_set)
    dataloaders = {'train': loader_train, 'val': loader_test}

    # Load model
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(NUM_CLASSES) # replace final layer to work with new dataset
    i3d = i3d.to(device=device) # move the model parameters to CPU/GPU

    # Define optimizer
    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    # Start training
    num_steps_per_update = 4 # accum gradient
    steps = 0
    while steps < max_steps: # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # set model to eval mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                inputs, _, class_idx = data # 2nd element of data tuple is audio
                inputs = inputs.permute(0, 4, 1, 2, 3) # Swap from BTHWC to BCTHW
                inputs = inputs.to(device=device, dtype=torch.float32) # model expects inputs of float32
                labels = torch.zeros([NUM_CLASSES]) # create one-hot tensor for label
                labels[class_idx] = 1 
                labels = labels.view(1, NUM_CLASSES, 1) # reshape to match model output
                labels = labels.to(device=device)

                # Forward pass
                t = inputs.shape[2]
                per_frame_logits = i3d(inputs)
                print('per_frame_logits shape = {}'.format(per_frame_logits.shape))
                per_frame_logits = F.interpolate(per_frame_logits, size=t, mode='linear') # upsample to match number of frames

                # Compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss

                # Compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss

                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        # save model
                        torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.

            if phase == 'val':
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))


if __name__ == '__main__':
    train(fold=1, use_gpu=True)
