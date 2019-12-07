import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from pytorch_i3d import *


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


class SIFE(nn.Module):
    """Scene Invariant Feature Learning network."""
    def __init__(self, backbone, num_features, num_actions, num_scenes, alpha=0.5, dropout_keep_prob=0.5):
        super(SIFE, self).__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        self.num_scenes = num_scenes
        self.alpha = alpha # constant that gradient reversal layer is multiplied by
        self.dropout_keep_prob = dropout_keep_prob

        # Backbone for feature extractor
        self.backbone = backbone

        # Network for action classification
        self.action_branch = nn.Sequential()
        self.action_branch.add_module('a_dropout', nn.Dropout(self.dropout_keep_prob))
        self.action_branch.add_module('a_unit3d', Unit3D(in_channels=384+384+128+128, 
                                                         output_channels=self.num_actions,
                                                         kernel_shape=[1, 1, 1],
                                                         padding=0,
                                                         activation_fn=None,
                                                         use_batch_norm=False,
                                                         use_bias=True,
                                                         name='logits'))

        # Network for scene classification
        self.scene_branch = nn.Sequential()
        self.scene_branch.add_module('s_fc1', nn.Linear(self.num_features, self.num_features))
        self.scene_branch.add_module('s_bn1', nn.BatchNorm1d(self.num_features))
        self.scene_branch.add_module('s_relu1', nn.ReLU(inplace=True))
        self.scene_branch.add_module('s_fc2', nn.Linear(self.num_features, self.num_features))
        self.scene_branch.add_module('s_bn2', nn.BatchNorm1d(self.num_features))
        self.scene_branch.add_module('s_relu2', nn.ReLU(inplace=True))
        self.scene_branch.add_module('s_fc3', nn.Linear(self.num_features, num_scenes))

    def forward(self, x):
        features = self.backbone.extract_features(x)
        
        action_logits = self.action_branch(features)
        action_logits = action_logits.squeeze(3).squeeze(3) # original i3d model performs this so copying it here as well

        reversed_features = ReverseLayerF.apply(features, self.alpha) # reverse the gradient
        reversed_features = reversed_features.squeeze() # scene_branch expects 1D vectors
        scene_logits = self.scene_branch(reversed_features)

        return action_logits, scene_logits
