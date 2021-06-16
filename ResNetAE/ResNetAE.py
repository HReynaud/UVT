from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

###############################################################################
# This code is directly extracted from https://github.com/farrell236/ResNetAE #
# Unused parts were removed                                                   #
###############################################################################


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):

        super(ResidualBlock, self).__init__()

        self.residual_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)


class ResNetEncoder(torch.nn.Module):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 input_ch=3,
                 z_dim=10,
                 bUseMultiResSkips=True):

        super(ResNetEncoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = torch.nn.ModuleList()
        self.res_blk_list = torch.nn.ModuleList()
        self.multi_res_skip_list = torch.nn.ModuleList()

        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_ch, out_channels=8,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
        )

        for i in range(n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (n_levels - i)

            self.res_blk_list.append(
                torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(n_filters_1, n_filters_2,
                                    kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_2),
                    torch.nn.ReLU(inplace=True),
                )
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters, kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        torch.nn.BatchNorm2d(self.max_filters),
                        torch.nn.ReLU(inplace=True),
                    )
                )

        self.output_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.max_filters, out_channels=z_dim,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(z_dim),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x = self.input_conv(x)

        skips = []
        for i in range(self.n_levels):
            x = self.res_blk_list[i](x)
            if self.bUseMultiResSkips:
                skips.append(self.multi_res_skip_list[i](x))
            x = self.conv_list[i](x)

        if self.bUseMultiResSkips:
            x = sum([x] + skips)

        x = self.output_conv(x)

        return x


class ResNetDecoder(torch.nn.Module):
    def __init__(self,
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=10,
                 output_channels=3,
                 bUseMultiResSkips=True):

        super(ResNetDecoder, self).__init__()

        self.max_filters = 2 ** (n_levels+3)
        self.n_levels = n_levels
        self.bUseMultiResSkips = bUseMultiResSkips

        self.conv_list = torch.nn.ModuleList()
        self.res_blk_list = torch.nn.ModuleList()
        self.multi_res_skip_list = torch.nn.ModuleList()

        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=z_dim, out_channels=self.max_filters,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(self.max_filters),
            torch.nn.ReLU(inplace=True),
        )

        for i in range(n_levels):
            n_filters_0 = 2 ** (self.n_levels - i + 3)
            n_filters_1 = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                torch.nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                                      for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(n_filters_0, n_filters_1,
                                             kernel_size=(2, 2), stride=(2, 2), padding=0),
                    torch.nn.BatchNorm2d(n_filters_1),
                    torch.nn.ReLU(inplace=True),
                )
            )

            if bUseMultiResSkips:
                self.multi_res_skip_list.append(
                    torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(in_channels=self.max_filters, out_channels=n_filters_1,
                                                 kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        torch.nn.BatchNorm2d(n_filters_1),
                        torch.nn.ReLU(inplace=True),
                    )
                )

        self.output_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=n_filters_1, out_channels=output_channels,
                            kernel_size=(3, 3), stride=(1, 1), padding=1),
            # torch.nn.BatchNorm2d(output_channels),
            # torch.nn.ReLU(inplace=True),
        )

    def forward(self, z):

        z = z_top = self.input_conv(z)

        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.bUseMultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        z = self.output_conv(z)

        return z


class ResNetAE(torch.nn.Module):
    def __init__(self,
                 input_shape=(256, 256, 3),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True):
        super(ResNetAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        image_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)

        self.fc1 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc2 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return torch.tanh(self.fc1(h.view(x.shape[0], self.z_dim * self.img_latent_dim * self.img_latent_dim)))

    def decode(self, z):
        h = self.decoder(self.fc2(z).view(-1, self.z_dim, self.img_latent_dim, self.img_latent_dim))
        return torch.sigmoid(h)

    def forward(self, x):
        return self.decode(self.encode(x))


class ResNetVAE(torch.nn.Module):
    def __init__(self,
                 input_shape=(256, 256, 3),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True):
        super(ResNetVAE, self).__init__()

        assert input_shape[0] == input_shape[1]
        image_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     input_ch=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     output_channels=image_channels, z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)

        # Assumes the input to be of shape 256x256
        self.fc21 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc22 = torch.nn.Linear(self.z_dim * self.img_latent_dim * self.img_latent_dim, bottleneck_dim)
        self.fc3 = torch.nn.Linear(bottleneck_dim, self.z_dim * self.img_latent_dim * self.img_latent_dim)

    def encode(self, x):
        h1 = self.encoder(x)
        return self.fc21(h1.view(-1, self.z_dim * self.img_latent_dim * self.img_latent_dim)), \
               self.fc22(h1.view(-1, self.z_dim * self.img_latent_dim * self.img_latent_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.decoder(self.fc3(z).view(-1, self.z_dim, self.img_latent_dim, self.img_latent_dim))
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == '__main__':

    encoder = ResNetEncoder()
    decoder = ResNetDecoder()

    test_input = torch.rand(10, 3, 256, 256)
    out = encoder(test_input)

    test_input = torch.rand(10, 10, 16, 16)
    out = decoder(test_input)

    a=1

    ae = ResNetAE()
    out = ae(torch.rand(10, 3, 256, 256))

    a=1

    vae = ResNetVAE()
    out = vae(torch.rand(10, 3, 256, 256))

    a=1