from torchvision.datasets.ucf101 import UCF101
from torch.utils.data import DataLoader

def load_ucf101():
    d = UCF101(root='/Users/rhsieh/Desktop/Stanford/CV_Research/ucf101/data',
               annotation_path='/Users/rhsieh/Desktop/Stanford/CV_Research/ucf101/ucfTrainTestlist',
               frames_per_clip=16,
               step_between_clips=16,
               fold=1,
               train=True)

    return d

if __name__ == '__main__':
    d = load_ucf()
    train_loader = DataLoader(d)

    print(len(train_loader))
    for i, (sample, label) in enumerate(train_loader):
        print('{}: shape={}, label={}'.format(i, sample.shape, label))
        if i == 10:
            break
