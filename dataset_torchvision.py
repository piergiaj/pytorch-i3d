import os
import videotransforms
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torchvision.datasets.ucf101 import UCF101
from torch.utils.data import DataLoader


def load_ucf101(root, annotation_path, frames_per_clip, step_between_clips, fold=1, train=True, transform=None):
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


def inspect_dataset(dataset, num_samples=5):
    fig = plt.figure()

    for i in num_samples:
        sample = dataset[i]
        print(i, sample.shape) # Check this

        ax = plt.subplot(1, num_samples, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.show()


def inspect_dataloader(dataloader, num_samples=5):
    fig = plt.figure()

    for i, data in enumerate(dataloader):
        pass


if __name__ == '__main__':
    root = os.path.join(os.getcwd(), 'data/ucf101/clips')
    annotation_path = os.path.join(os.getcwd(), 'data/ucf101/ucfTrainTestlist')

    d = UCF101(root=root, annotation_path=annotation_path, frames_per_clip=16, step_between_clips=16, fold=1, train=False, transform=None)


    # transform = T.Compose([videotransforms.RandomCrop(224)])

    # train_loader = load_ucf101(root,
    #                            annotation_path,
    #                            transform=transform)
    # Each item in train_loader is a tuple of length 3:
    # (Tensor[T, H, W, C]) audio: (Tensor[K, L]), label (int))