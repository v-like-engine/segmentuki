import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torch

from models.SPADE.normalization_layers import SPADE
from models.pix2pixHD.networks import GANLoss, MultiscaleDiscriminator


class SPADEResnetBlock(nn.Module):
    """
    ResnetBlock with SPADE normalization. With the same notation used in the original code made by NVIDIA.
    It differs from the ResNet block of pix2pixHD in that
    it takes in the segmentation map as input, learns the skip connection if necessary,
    and applies normalization first and then convolution.
    """
    def __init__(self, fin, fout, label_nc=150):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.convolution0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.convolution1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.convolutionS = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        self.normalization0 = SPADE('batch', 5, fin, label_nc)
        self.normalization1 = SPADE('batch', 5, fmiddle, label_nc)
        if self.learned_shortcut:
            self.normalizationS = SPADE('batch', 5, fin, label_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.convolution0(self.actvn(self.normalization0(x, seg)))
        dx = self.convolution1(self.actvn(self.normalization1(dx, seg)))

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


class SPADEGenerator(nn.Module):
    def __init__(self, input_channels, output_channels, ngf, semantic_nc=150, use_vae=False, z_dim=None, device='cpu'):
        super(SPADEGenerator, self).__init__()
        self.sw = 128 // (2**5)
        self.sh = self.sw
        if use_vae:
            # If VAE, we utilize sampling from random Z vector with dimension z_dim
            self.fc = nn.Linear(z_dim, 16 * ngf * self.sw * self.sh)
        else:
            # else, we work with segmentation map from the beginning
            self.fc = nn.Conv2d(semantic_nc, 16 * ngf, 3, padding=1)
        self.device = device

        self.head_0 = SPADEResnetBlock(16 * ngf, 16 * ngf)

        self.G_middle_0 = SPADEResnetBlock(16 * ngf, 16 * ngf)
        self.G_middle_1 = SPADEResnetBlock(16 * ngf, 16 * ngf)

        self.up_0 = SPADEResnetBlock(16 * ngf, 8 * ngf)
        self.up_1 = SPADEResnetBlock(8 * ngf, 4 * ngf)
        self.up_2 = SPADEResnetBlock(4 * ngf, 2 * ngf)
        self.up_3 = SPADEResnetBlock(2 * ngf, 1 * ngf)
        self.conv_img = nn.Conv2d(ngf, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, seg):
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)
        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class Pix2PixHDwithSPADE(nn.Module):
    def __init__(self, lr_g=0.001, lr_d=0.004, beta1=0.9, beta2=0.999, semantic_nc=3, device='cuda'):
        """
        :param lr: learning rate for both generator and discriminator
        :param beta1 and beta2: coefficients used for computing running averages of gradient and its square
        """
        super(Pix2PixHDwithSPADE, self).__init__()
        self.generator = SPADEGenerator(3, 3, 64, semantic_nc, device=device).to(device)
        self.discriminator = MultiscaleDiscriminator(3, 64, 2, 2, device).to(device)

        self.criterion_gan = GANLoss().to(device)

        # generator optimizer
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(beta1, beta2))

        # discriminator optimizer
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

    def forward(self, label, image):
        fake_image = self.generator.forward(label)
        # fake_input_concat = torch.cat((label, fake_image.detach()), dim=1)
        pred_fake = self.discriminator.forward(fake_image)
        loss_D_fake = self.criterion_gan(pred_fake, False)

        # real_input_concat = torch.cat((label, image.detach()), dim=1)
        pred_real = self.discriminator.forward(image)
        loss_D_real = self.criterion_gan(pred_real, True)

        fake_image = self.generator.forward(label)
        # fake_input_concat = torch.cat((label, fake_image.detach()), dim=1)
        pred_fake = self.discriminator.forward(fake_image)
        loss_G = self.criterion_gan(pred_fake, True)

        return [loss_G, loss_D_real, loss_D_fake]

    def inference(self, label):
        with torch.no_grad():
            return self.generator.forward(label)

