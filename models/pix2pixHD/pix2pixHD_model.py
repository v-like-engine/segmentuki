from models.pix2pixHD.networks import Generator, MultiscaleDiscriminator, GANLoss
# from networks import Generator, MultiscaleDiscriminator, GANLoss
import torch.nn as nn
import torch


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

        self.criterion_gan = GANLoss()

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

        return [[loss_G, loss_D_real, loss_D_fake], None]
