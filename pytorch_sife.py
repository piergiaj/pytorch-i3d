import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from pytorch_i3d import InceptionI3d


## ReverseLayerF is copied from https://github.com/jindongwang/transferlearning/tree/master/code/deep/DANN(RevGrad)
## Original paper: Ganin Y, Lempitsky V. Unsupervised domain adaptation by backpropagation. ICML 2015.
class ReverseLayerF(Function):
    """Reverse the gradients."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class ScenePredictor(nn.Module):
    """Scene prediction branch. Inputs are features from the main network."""
    def __init__(self, num_features, num_scene_classes):
        super().__init__()

        self.fc1 = nn.Linear(in_features=num_features, out_features=num_features)
        self.bn1 = nn.BatchNorm1d(num_features=num_features)
        self.fc2 = nn.Linear(in_features=num_features, out_features=num_features)
        self.bn2 = nn.BatchNorm1d(num_features=num_features)
        self.fc3 = nn.Linear(in_features=num_features, out_features=num_scene_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.fc3(x)

        return x


class SIFE(InceptionI3d):
    """Scene Invariant Feature Learning network."""
    def __init__(self, num_action_classes=400, spatial_squeeze=True, final_endpoint='Logits', 
                 name='SIFE_i3d', in_channels=3, dropout_keep_prob=0.5, 
                 num_features, num_scene_classes):
        super().__init__(num_action_classes=400, spatial_squeeze=True, final_endpoint='Logits', 
                         name='SIFE_i3d', in_channels=3, dropout_keep_prob=0.5)

        self.Scene_Branch = ScenePredictor(num_features, num_scene_classes)
        # TODO: might have to change variable 'name' from the base class

    def forward(self, x):
        # super() provides access to methods of the base class which have been overridden in the subclass 
        action_logits = super().forward(x)
        features = self.extract_features(x)
        reversed_features = ReverseLayerF.apply(features)
        scene_logits = self.Scene_Branch(reversed_features)

        return action_logits, scene_logits
