import os
from torchvision.datasets.ucf101 import UCF101
from torch.utils.data import DataLoader

def load_ucf101(root='data/ucf101/video', annotation_path='data/ucf101/ucfTrainTestlist'):
    root = os.path.join(os.getcwd(), root)
    annotation_path = os.path.join(os.getcwd(), annotation_path)

    d = UCF101(root=root,
               annotation_path=annotation_path,
               frames_per_clip=16,
               step_between_clips=16,
               fold=1,
               train=True)

    return d


if __name__ == '__main__':
    d = load_ucf101()
    train_loader = DataLoader(d)

    print(len(train_loader))
    for i, (sample, label) in enumerate(train_loader):
        print('{}: shape={}, label={}'.format(i, sample.shape, label))
        if i == 10:
            break
