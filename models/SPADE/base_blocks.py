import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BaseConv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StandardConvBaseModule(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.norm_nc = norm_nc
        self.label_nc = label_nc
        self.conv1 = BaseConv2dBlock(label_nc + norm_nc, norm_nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = BaseConv2dBlock(norm_nc, norm_nc, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = BaseConv2dBlock(norm_nc, norm_nc, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, segmap):
        # Normalize the feature maps using the segmentation map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        x = torch.cat((x, segmap), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SPADE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(SPADE, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x):
        # Generate the spatially-adaptive normalization parameters
        gamma, beta = self.conv(x).chunk(2, dim=1)

        # Apply the normalization
        out = F.instance_norm(x, gamma, beta)

        return out


class SPADEConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SPADEConv2dBlock, self).__init__()
        self.sn = SPADE(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.sn(x)
        x = self.relu(x)
        return x
