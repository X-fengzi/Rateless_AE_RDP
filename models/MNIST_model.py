import torch.nn as nn
import torch

from utils import generate_centers
import models.base_model as base_model
import torch.nn.functional as F



class EncoderMNIST(nn.Module):
    def __init__(self, args):
        super(EncoderMNIST, self).__init__()
        self.args = args
        self.n_channel = args.n_channel
        self.latent_dim = args.latent_dim
        self.quantize_latents = args.quantize
        self.stochastic = args.stochastic
        self.ls = args.enc_layer_scale
        self.input_size = args.input_size
        self.n_class = args.n_class

        self.use_si = args.use_si
        self.only_si = args.only_si

        # add one-hot dims if use si
        if self.only_si:
            self.en_in_size = args.n_class
        else:
            self.en_in_size = args.input_size + args.n_class if args.use_si else args.input_size

        L = args.L
        q_limits = args.limits

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = base_model.Quantizer(centers=centers)
        if self.stochastic:
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)
            else:
                raise ValueError('Quant. disabled')

        final_layer_width = int(self.ls*128)
        self.net = nn.Sequential(
            nn.Linear(self.en_in_size, 4*final_layer_width),
            nn.BatchNorm1d(4*final_layer_width),
            nn.LeakyReLU(),
            nn.Linear(4*final_layer_width, 2*final_layer_width),
            nn.BatchNorm1d(2*final_layer_width),
            nn.LeakyReLU(),
            nn.Linear(2*final_layer_width, final_layer_width),
            nn.BatchNorm1d(final_layer_width),
            nn.LeakyReLU(),
            nn.Linear(final_layer_width, final_layer_width),
            nn.BatchNorm1d(final_layer_width),
            nn.LeakyReLU(),
            nn.Linear(final_layer_width, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.Tanh()
        )
        self.final_layer_width = final_layer_width

    def add_si(self,x,y):
        y = F.one_hot(y, num_classes=self.n_class)
        # [bsize, n_class]
        if self.only_si:
            x = y.float()
        else:
            x = torch.concat([x,y],dim=1)
            # [bsize, w*h + n_class]
        return x

    def forward(self, x, u, y):
        x = x.view(-1, self.input_size)
        # x: [bsize, 28*28]
        # y: [bsize]

        if self.use_si:
            x = self.add_si(x, y)
        x = self.net(x)

        # add randomness at quantizer
        if self.stochastic:
            x = x + u
        if self.quantize_latents and (not self.only_si):
            x = self.q(x)

        return x


class DecoderMNIST(nn.Module):
    def __init__(self, args):
        super(DecoderMNIST, self).__init__()
        self.latent_dim = args.latent_dim
        self.output_size = args.input_size
        self.use_si = args.use_si
        self.n_class = args.n_class

        self.de_in_size = args.latent_dim + args.n_class if args.use_si else args.latent_dim

        self.net = nn.Sequential(
            nn.Linear(self.de_in_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        if self.output_size == 784:
            self.deconvolve = nn.Sequential(
                nn.ConvTranspose2d(32, 64, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 128, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 1, kernel_size=4, stride=1, padding=0),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f'No deconvolution defined for output size of {self.output_size}.')

    def add_si(self,x,y):
        y = F.one_hot(y, num_classes=self.n_class)
        x = torch.concat([x, y], dim=1)
        return x

    def forward(self, x, u, y):
        x = x - u
        if self.use_si:
            x = self.add_si(x, y)
        x = self.net(x)
        x = x.view(-1, 32, 4, 4)
        x = self.deconvolve(x)

        return x


class DiscriminatorMNIST(nn.Module):
    def __init__(self, args):
        super(DiscriminatorMNIST, self).__init__()
        self.n_channel = args.n_channel

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 4096)
        x = self.fc(x)
        return x



