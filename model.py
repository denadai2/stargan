import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', pre_activation=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.pre_activation = pre_activation
        self.padding = padding
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if pre_activation:
            norm_dim = input_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim, affine=True, track_running_stats=True)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'adain_affine':
            self.norm = AdaptiveInstanceNormWithAffineTransform(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=False)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        if self.pre_activation:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)

        if self.padding != 0:
            x = self.pad(x)
        x = self.conv(x)

        if not self.pre_activation:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class PreActivationResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim, norm, padding=1):
        super(PreActivationResidualBlock, self).__init__()
        self.main = nn.Sequential(
            Conv2dBlock(dim, dim, kernel_size=3, stride=1, padding=padding, activation='relu', pad_type='zero',
                        norm=norm,
                        pre_activation=True),
            Conv2dBlock(dim, dim, kernel_size=3, stride=1, padding=1, activation='relu', pad_type='zero',
                        norm=norm,
                        pre_activation=True))

    def forward(self, x):
        return x + self.main(x)



class PreActivationResidualBlockWithAvgPool(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, norm, kernel_size=3, padding=1):
        super(PreActivationResidualBlockWithAvgPool, self).__init__()
        self.main = nn.Sequential(
            Conv2dBlock(dim_in, dim_out, kernel_size=kernel_size, stride=1, padding=padding, activation='relu', pad_type='zero',
                        norm=norm,
                        pre_activation=True),
            Conv2dBlock(dim_out, dim_out, kernel_size=kernel_size, stride=1, padding=padding, activation='relu', pad_type='zero',
                        norm=norm,
                        pre_activation=True))
        self.residual = Conv2dBlock(dim_in, dim_out, kernel_size, 1, 1, pad_type='zero')
        self.avg_pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.avg_pool(self.residual(x) + self.main(x))


class PreActivationResidualBlockWithUpsample(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, norm, scale_factor=2, padding=1):
        super(PreActivationResidualBlockWithUpsample, self).__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor)

        self.main = nn.Sequential(
            Conv2dBlock(dim_in, dim_out, kernel_size=3, stride=1, padding=padding, activation='relu', pad_type='zero',
                        norm=norm,
                        pre_activation=True),
            Conv2dBlock(dim_out, dim_out, kernel_size=3, stride=1, padding=1, activation='relu', pad_type='zero',
                        norm=norm,
                        pre_activation=True))
        self.residual = Conv2dBlock(dim_in, dim_out, 3, 1, 1, pad_type='zero')

    def forward(self, x):
        x = self.upsample(x)
        return self.residual(x) + self.main(x)


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None

        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


# Adain with affine transformation
class AffineLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.bias.data[:out_features // 2] = 1  # init gamma close to 1
        self.bias.data[out_features // 2:] = 0  # init beta close to 0
        self.is_affine = True
        self.nonlinearity = 'none'


class AdaptiveInstanceNormWithAffineTransform(nn.Module):
    def __init__(self, in_channel, style_dim=64):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = AffineLinear(style_dim, in_channel * 2)

    def forward(self, input):
        out = self.norm(input)
        out = self.gamma * out + self.beta
        return out


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, num_bottleneck=2, num_down_sample=4):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=1, stride=1, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(num_down_sample):
            layers.append(PreActivationResidualBlockWithAvgPool(dim_in=curr_dim, dim_out=curr_dim * 2, norm='in'))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(num_bottleneck):
            layers.append(PreActivationResidualBlock(dim=curr_dim, norm='in'))
        for i in range(num_bottleneck):
            layers.append(PreActivationResidualBlock(dim=curr_dim, norm='adain_affine'))

        # Up-sampling layers.
        for i in range(num_down_sample):
            layers.append(PreActivationResidualBlockWithUpsample(dim_in=curr_dim, dim_out=curr_dim // 2,
                                                                 norm='adain_affine', scale_factor=2))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=1, stride=1, padding=0, bias=False))
        self.main = nn.Sequential(*layers)

    def assign_adain_params_features(self, adain_params):
        for module in self.main.modules():  # self.dec
            if module.__class__.__name__ == 'AdaptiveInstanceNormWithAffineTransform':
                affine_transformed = module.style(adain_params).view(adain_params.size(0), -1, 1, 1)
                module.gamma, module.beta = affine_transformed.chunk(2, 1)

    def forward(self, x, style):
        self.assign_adain_params_features(style)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, conv_dim=16, n_domains=5, out_dim=1, repeat_num=6):
        super(Discriminator, self).__init__()
        self.n_domains = n_domains
        self.out_dim = out_dim

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=1, stride=1, padding=0))
        curr_dim = conv_dim
        for i in range(0, repeat_num):
            next_dim = curr_dim
            if i < repeat_num - 1:
                next_dim = curr_dim*2
            layers.append(PreActivationResidualBlockWithAvgPool(dim_in=curr_dim, dim_out=next_dim, norm='none', padding=1))

            curr_dim = next_dim

        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=2, stride=1, padding=0))
        layers.append(nn.LeakyReLU(0.01))

        layers.append(Conv2dBlock(curr_dim, out_dim * n_domains,  kernel_size=1, stride=1, padding=0, norm='none', activation='none'))

        self.main = nn.Sequential(*layers)
        
    def forward(self, x, domain):
        batch_size = domain.size(0)

        h = self.main(x)
        h = h.view(batch_size, self.n_domains, self.out_dim)

        domain = domain.repeat(1, 1, h.size(-1))
        return torch.gather(h, 1, domain.long())


class Mapping(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, latent_dim=16, linear_dim=64, n_domains=5, out_dim=64, repeat_num=6):
        super(Mapping, self).__init__()
        layers = []

        curr_dim = latent_dim
        for i in range(1, repeat_num):
            layers.append(nn.Linear(curr_dim, linear_dim, bias=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = linear_dim
        layers.append(nn.Linear(linear_dim, out_dim * n_domains, bias=True))

        self.n_domains = n_domains
        self.out_dim = out_dim
        self.main = nn.Sequential(*layers)

    def forward(self, x, domain):
        batch_size = domain.size(0)
        h = self.main(x)
        h = h.repeat(batch_size, 1).view(batch_size, self.n_domains, self.out_dim)

        domain = domain.repeat(1, 1, h.size(-1))
        return torch.gather(h, 1, domain.long())
