
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """
    SPADE layer code with the same notation as in the original code made by NVIDIA
    """
    def __init__(self, param_free_norm_type, ks, norm_nc, label_nc, nhidden=128):
        """

        :param param_free_norm_type:
        :param ks:
        :param norm_nc:
        :param label_nc: number of labels in the label images, they will be split into separate images and concatenated
            in the channels dimension
        :param nhidden:
        """

        super().__init__()

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = (self.
                shared(segmap))
        gamma = (self.
                 gamma(actv))
        beta = (self.
                beta(actv))

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
