import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torch

from models.SPADE.normalization_layers import SPADE
from models.pix2pixHD.networks import GANLoss


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
    def __init__(self, input_channels, output_channels, ngf, num_down_sampling_global, num_res_blocks_global,
                 num_local_enhancer_layers, num_res_blocks_local, device='cpu'):
        super(SPADEGenerator, self).__init__()
        self.num_local_enhancer_layers = num_local_enhancer_layers
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

    def forward(self, x, seg):
        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
                self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class GlobalGenerator(nn.Module):
    """
    The global generator network operates at a resolution of 1024 × 512.
    It consists of 3 components: a convolutional front-end G^F_1, a set of residual blocks G^R_1,
    and a transposed convolutional back-end G^B_1. A semantic label map of resolution
    1024×512 is passed through the 3 components sequentially to output an image of resolution 1024 × 512.

    Links:
    [1] GlobalGenerator
    https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf
    https://github.com/aliissaoui/Perceptual-loss-for-style-transfer/
    """

    def __init__(self, input_channels, output_channels, ngf, num_down_sampling, num_res_blocks):
        """
        :param input_channels: number of channels in the input image
        :param output_channels: number of channels in the output image
        :param ngf: number of the filters
        :param num_down_sampling: number of down sampling operation
        :param num_res_blocks: number of residual blocks
        """
        super(GlobalGenerator, self).__init__()

        padding_layer = nn.ReflectionPad2d(3)  # some padding
        norm_layer = nn.BatchNorm2d  # some normalization
        activation = nn.ReLU(True)  # some activation function
        model_layers = [padding_layer, nn.Conv2d(input_channels, ngf, kernel_size=7), norm_layer(ngf), activation]

        # G_F or encoder
        for i in range(num_down_sampling):
            mult = 2 ** i
            model_layers += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                             norm_layer(ngf * mult * 2), activation]

        # G_R or resnet block
        mult = 2 ** num_down_sampling
        for i in range(num_res_blocks):
            model_layers += [ResidualBlock(ngf * mult)]

        # G_B or decoder
        for i in range(num_down_sampling - 1, -1, -1):
            mult = 2 ** i
            model_layers += [
                nn.ConvTranspose2d(ngf * mult * 2, ngf * mult, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf * mult), activation]

        model_layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_channels, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model_layers)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):

    def __init__(self, dim):
        """
        :param dim: size of hidden dimension
        """
        super(ResidualBlock, self).__init__()

        padding_layer = nn.ReflectionPad2d(1)  # some padding
        norm_layer = nn.BatchNorm2d  # some normalization
        activation = nn.ReLU(True)  # some activation function

        self.model = nn.Sequential(
            padding_layer,
            nn.Conv2d(dim, dim, kernel_size=3),
            norm_layer(dim),
            activation,
            padding_layer,
            nn.Conv2d(dim, dim, kernel_size=3),
            norm_layer(dim)
        )

    def forward(self, x):
        return self.model(x) + x


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_channels, ndf, num_layers, num_discriminators, device='cpi'):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_discriminators = num_discriminators
        self.num_layers = num_layers

        self.discriminators = nn.ModuleList()

        for i in range(num_discriminators):
            self.discriminators.append(NLayerDiscriminator(input_channels, ndf, num_layers).to(device).model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=(1, 1), count_include_pad=False)

    def forward(self, x):
        result = []

        x_downsampled = x

        for i in range(self.num_discriminators):
            result.append(self.discriminators[i](x_downsampled))
            if i != self.num_discriminators - 1:
                x_downsampled = self.downsample(x)
        return result


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_channels, ndf=64, num_layers=3):
        super(NLayerDiscriminator, self).__init__()
        self.num_layers = num_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_channels, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, num_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm2d(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        sequence += [[nn.Sigmoid()]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

    def forward(self, x):
        return self.model(x)


class Pix2PixHD(nn.Module):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, device='cuda'):
        """
        TODO: add different lr for generator and discriminator
        :param lr: learning rate for both generator and discriminator
        :param beta1 and beta2: coefficients used for computing running averages of gradient and its square
        """
        super(Pix2PixHD, self).__init__()
        self.generator = Generator(3, 3, 64, 1, 6, 1, 3, device).to(device)
        self.discriminator = MultiscaleDiscriminator(3, 64, 2, 2, device).to(device)

        self.criterion_gan = GANLoss().to(device)

        # generator optimizer
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))

        # discriminator optimizer
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

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

