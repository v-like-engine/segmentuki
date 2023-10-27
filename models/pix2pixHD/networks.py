import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, ngf, num_down_sampling_global, num_res_blocks_global,
                 num_local_enhancer_layers, num_res_blocks_local):
        super(Generator, self).__init__()
        self.num_local_enhancer_layers = num_local_enhancer_layers

        # global generator
        ngf_global = ngf * (2 ** self.num_local_enhancer_layers)
        model_global = GlobalGenerator(input_channels, output_channels, ngf_global, num_down_sampling_global,
                                       num_res_blocks_global).model
        model_global = [model_global[i] for i in range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        # local enhancer layers
        padding_layer = nn.ReflectionPad2d(3)  # some padding
        norm_layer = nn.BatchNorm2d  # some normalization
        activation = nn.ReLU(True)  # some activation function
        self.models_downsample = [0]
        self.models_upsample = [0]

        for n in range(1, num_local_enhancer_layers + 1):
            # downsample
            ngf_global = ngf * (2 ** (num_local_enhancer_layers - n))
            model_downsample = [padding_layer, nn.Conv2d(input_channels, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), activation,
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), activation]
            # residual blocks
            model_upsample = []
            for i in range(num_res_blocks_local):
                model_upsample += [ResidualBlock(ngf_global * 2)]

            # upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            # final convolution
            if n == num_local_enhancer_layers:
                model_upsample += [padding_layer, nn.Conv2d(ngf, output_channels, kernel_size=7, padding=0),
                                   nn.Tanh()]

            self.models_downsample.append(nn.Sequential(*model_downsample))
            self.models_upsample.append(nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=(1, 1), count_include_pad=False)

    def forward(self, x):
        x_downsampled = [x]
        for i in range(self.num_local_enhancer_layers):
            x_downsampled.append(self.downsample(x_downsampled[-1]))

        output_prev = self.model(x_downsampled[-1])

        for num_local_enhancer_layers in range(1, self.num_local_enhancer_layers + 1):
            input_i = x_downsampled[self.num_local_enhancer_layers - num_local_enhancer_layers]
            output_prev = self.models_upsample[num_local_enhancer_layers](
                self.models_downsample[num_local_enhancer_layers](input_i) + output_prev)
        return output_prev


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
    def __init__(self, input_channels, ndf, num_layers, num_discriminators):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_discriminators = num_discriminators
        self.num_layers = num_layers

        self.discriminators = nn.ModuleList()

        for i in range(num_discriminators):
            self.discriminators.append(NLayerDiscriminator(input_channels, ndf, num_layers).model)
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
