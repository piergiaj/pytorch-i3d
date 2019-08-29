import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from ucf101 import UCF101
from spatial_transforms import Compose, ToTensor, Scale


def inspect_dataset(dataset, num_samples=5):
    fig = plt.figure()

    for i in range(num_samples):
        data = dataset[i]
        clip = data[0]
        print(i, clip.shape) # Check this

        ax = plt.subplot(1, num_samples, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.show()


def inspect_dataloader(dataloader, num_samples=5):
    fig = plt.figure()

    for i, data in enumerate(dataloader):
        clip = data[0]
        # TODO: Finish writing this function


if __name__ == '__main__':
    FOLD = 1
    BATCH_SIZE = 8
    NUM_WORKERS = 1
    SHUFFLE = True

    video_path = '/vision/u/rhsieh91/UCF101/jpg'
    annotation_path = '/vision/u/rhsieh91/UCF101/ucfTrainTestlist/ucf101_0' + str(FOLD) + '.json'

    SPATIAL_TRANSFORM = Compose([
        Scale((224, 224)),
        ToTensor()
        ])

    d_train = UCF101(video_path,
                 annotation_path,
                 subset='training',
                 n_samples_for_each_video=4,
                 spatial_transform=SPATIAL_TRANSFORM)
    inspect_dataset(d_train)

    train_loader = DataLoader(d_train, 
                          batch_size=BATCH_SIZE,
                          shuffle=SHUFFLE, 
                          num_workers=NUM_WORKERS,
                          pin_memory=True)
    # inspect_dataloader(train_loader)
