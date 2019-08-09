import os
import torchvision.transforms as T

from torchvision.datasets.ucf101 import UCF101
from torch.utils.data import DataLoader



def load_ucf101(root, annotation_path):
    d = UCF101(root=root,
               annotation_path=annotation_path,
               frames_per_clip=16,
               step_between_clips=16,
               fold=1,
               train=True)

    return d


if __name__ == '__main__':
    root = os.path.join(os.getcwd(), 'data/ucf101/clips')
    annotation_path = os.path.join(os.getcwd(), 'data/ucf101/ucfTrainTestlist')

    d = load_ucf101(root, annotation_path)
    train_loader = DataLoader(d)

    print(len(train_loader))
    for i, (sample, label) in enumerate(train_loader):
        print('{}: shape={}, label={}'.format(i, sample.shape, label))
        if i == 10:
            break
