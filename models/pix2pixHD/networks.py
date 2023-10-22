import torch
import torch.nn as nn


class GlobalGenerator(torch.nn.Module):
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

    def __init__(self, input_channels, output_channels, ngf, num_down_sampling):
        """

        :param input_channels: number of channels in the input image
        :param output_channels: number of channels in the output image
        :param ngf: number of the filters
        :param num_down_sampling: number of down sampling operation
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
        ...

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
