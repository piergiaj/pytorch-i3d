import os
import torchvision.transforms as T
import videotransforms

from torchvision.datasets.ucf101 import UCF101
from torch.utils.data import DataLoader



def load_ucf101(root, annotation_path, frames_per_clip=16, step_between_clips=16, fold=1):
    d = UCF101(root=root,
               annotation_path=annotation_path,
               frames_per_clip=frames_per_clip,
               step_between_clips=step_between_clips,
               fold=fold,
               train=True,
               transform=T.Compose([videotransforms.RandomCrop(224)]))

    return d


if __name__ == '__main__':
    root = os.path.join(os.getcwd(), 'data/ucf101/clips')
    annotation_path = os.path.join(os.getcwd(), 'data/ucf101/ucfTrainTestlist')

    d = load_ucf101(root, annotation_path)
    train_loader = DataLoader(d) 
    # each item in train_loader is a tuple of length 3
    # video: (Tensor[T, H, W, C]) audio: (Tensor[K, L]), label (int)
    # model expects BCTHW
