import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from models.SPADE.normalization_layers import SPADE


class SPADEResnetBlock(nn.Module):
    """
    ResnetBlock with SPADE normalization. With the same notation used in the original code made by NVIDIA.
    It differs from the ResNet block of pix2pixHD in that
    it takes in the segmentation map as input, learns the skip connection if necessary,
    and applies normalization first and then convolution.
    """
    def __init__(self, fin, fout, spectral=False):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.convolution0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.convoution1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.convolutionS = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if spectral:
            self.convolution0 = spectral_norm(self.convolution0)
            self.convoution1 = spectral_norm(self.convoution1)
            if self.learned_shortcut:
                self.convolutionS = spectral_norm(self.convolutionS)

        # define normalization layers
        self.normalization0 = SPADE('batch', 5, fin, fin)
        self.normalization1 = SPADE('batch', 5, fmiddle, fmiddle)
        if self.learned_shortcut:
            self.normalizationS = SPADE('batch', 5, fin, fin)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.convolution0(self.actvn(self.normalization0(x, seg)))
        dx = self.convoution1(self.actvn(self.normalization1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.convolutionS(self.normalizationS(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
