import os
import videotransforms
import torchvision.transforms as T

from torchvision.datasets.ucf101 import UCF101
from torch.utils.data import DataLoader


def load_ucf101(root, annotation_path, frames_per_clip, step_between_clips,
                data_loader, fold=1, train=True, transform=None):
    '''
    Helper function for loading UCF101 dataset into a Pytorch DataLoader object.
    '''
    d = UCF101(root=root,
               annotation_path=annotation_path,
               frames_per_clip=frames_per_clip,
               step_between_clips=step_between_clips,
               fold=fold,
               train=train,
               transform=transform)
    d_loader = DataLoader(d, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)

    return d_loader


def inspect_data():
    # TODO: Function to view select samples from dataset and make sure it is correct
    pass


if __name__ == '__main__':
    root = os.path.join(os.getcwd(), 'data/ucf101/clips')
    annotation_path = os.path.join(os.getcwd(), 'data/ucf101/ucfTrainTestlist')

    transform = T.Compose([videotransforms.RandomCrop(224)])

    train_loader = load_ucf101(root,
                               annotation_path,
                               transform=transform)
    # Each item in train_loader is a tuple of length 3:
    # (Tensor[T, H, W, C]) audio: (Tensor[K, L]), label (int))