import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable

from dataset import TSNDataSet
from pytorch_i3d import InceptionI3d


NUM_CLASSES = 101


def train(init_lr=0.1, max_steps=64e3, save_model=''):
    """
    Load pretrained I3D model and finetune on new dataset.
    """
    # Load data
    ucf101_train1 = TSNDataSet(root_path='', 
                               list_file='data/trainlist01.txt',
                               num_segments=3,
                               modality='RGB',
                               image_tmpl='image_{:04d}.jpg')
    ucf101_test1 = TSNDataSet(root_path='',
                              list_file='data/testlist01.txt',
                              num_segments=3,
                              modality='RGB',
                              image_tmpl='image_{:04d}.jpg')

    train_dataloader = DataLoader(ucf101_train1)
    test_dataloader = DataLoader(ucf101_test1)
    dataloaders = {'train': train_dataloader, 'val': test_dataloader}

    # Load model
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))

    # Replace final layer to work with new dataset
    i3d.replace_logits(NUM_CLASSES)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # # wrap them in Variable
                # inputs = Variable(inputs.cuda())
                # t = inputs.size(2)
                # labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data[0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data[0]

                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.data[0]
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
    train()
