from copy import deepcopy

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.MNIST_model import EncoderMNIST,DecoderMNIST,DiscriminatorMNIST


def Encoder(args):
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        baseclass = EncoderMNIST
    else:
        raise ValueError('Unknown dataset')

    class Encoder(baseclass):
        def __init__(self, args):
            super().__init__(args)

    return Encoder(args)


def Decoder(args):
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        baseclass = DecoderMNIST
    else:
        raise ValueError('Unknown dataset')

    class Decoder(baseclass):
        def __init__(self, args):
            super().__init__(args)

    return Decoder(args)


def Discriminator(args):
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        baseclass = DiscriminatorMNIST
    else:
        raise ValueError('Unknown dataset')

    class Discriminator(baseclass):
        def __init__(self, args):
            super().__init__(args)

    return Discriminator(args)


class Quantizer(nn.Module):
    def __init__(self, centers=[-1.0, 1.0], sigma=1.0):
        super(Quantizer, self).__init__()
        self.centers = centers
        self.sigma = sigma

    def forward(self, x):
        centers = x.data.new(self.centers)
        xsize = list(x.size())

        # Compute differentiable soft quantized version
        x = x.view(*(xsize + [1]))
        level_var = Variable(centers, requires_grad=False)
        # dist = torch.pow(x-level_var, 2)
        dist = torch.abs(x-level_var)
        output = torch.sum(level_var * nn.functional.softmax(-self.sigma*dist, dim=-1), dim=-1)

        # Compute hard quantization (invisible to autograd)
        _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
        for _ in range(len(xsize)): centers.unsqueeze(0) # in-place error
        centers = centers.expand(*(xsize + [len(self.centers)]))

        # Compute hard quantization (invisible to autograd)
        # _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
        # _centers = centers.clone()
        # for _ in range(len(xsize)): _centers.unsqueeze_(0) # in-place error
        # _centers = _centers.expand(*(xsize + [len(self.centers)]))

        quant = centers.gather(-1, symbols.long()).squeeze_(dim=-1)

        # Replace activations in soft variable with hard quantized version
        output.data = quant

        return output

class Dropout_rateless(nn.Module):

    def __init__(self, p=0.5, mode='uniform', distribution = None):
        super(Dropout_rateless, self).__init__()
        if p < 0 or p > 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p
        self.kp = 1 - p
        self.mode = mode
        self.f = distribution
        
    def forward(self, x):
        if self.training and self.p != 0:
        	# 生成mask矩阵。
        	# torch.rand_like：生成和x相同尺寸的张量，取值在[0,1)之间均匀分布。
            if self.mode ==  'uniform':
                mask = (torch.rand_like(x) < self.kp)
            else:
                x_shape = x.shape
                dim = x_shape[-1]
                expect_num = 0
                for i in range(dim):
                    expect_num += self.f(i/dim)
                norm_factor = dim*self.p/expect_num
                mask_p = torch.zeros_like(x)
                for j in range(dim):
                    mask_p[...,j] = norm_factor*self.f(j/dim)
                mask = torch.bernoulli(1-mask_p)
            # print(mask_p[0,:])
            # print(mask[0,:])
            # exit()
            # 先用mask矩阵对x矩阵进行处理，再除以1 - p（保留概率），即上述所说的反向DropOut操作，不需要在测试集上再缩放。
            return x * mask / self.kp
        else:
            return x
