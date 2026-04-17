# CMU 18-780/6 Homework 4
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
# For the assignment, you are asked to create the architectures of these
# three networks by filling in the __init__ and forward methods in the
# DCGenerator, DCDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn

def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Create a convolutional layer, with optional normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):

    def __init__(self, noise_size, conv_dim=64):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # input:  16x100x1x1
        # output: 16x3x64x64

        # the first layer (up conv1) it is better to directly apply convolution
        #   layer without any upsampling to get 4x4 output.
        # DCGAN: https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        # nn.Conv2d: https://www.google.com/search?client=safari&rls=en&q=nn.Conv2d&ie=UTF-8&oe=UTF-8
        # 100x1x1 to 256x4x4
        self.up_conv1 = nn.Sequential(nn.Conv2d(noise_size, conv_dim * 8, kernel_size=4, stride=1, padding=3, bias=False),
                        nn.BatchNorm2d(conv_dim * 8),
                        nn.ReLU(True))

        # 256x4x4 to 128x8x8
        self.up_conv2 = up_conv(conv_dim * 8, conv_dim * 4, kernel_size=3, norm='batch', activ='relu')

        # 128x8x8 to 64x16x16
        self.up_conv3 = up_conv(conv_dim * 4, conv_dim * 2, kernel_size=3, norm='batch', activ='relu')


        # 64x16x16 to 32x32x32
        self.up_conv4 = up_conv(conv_dim * 2, conv_dim * 1, kernel_size=3, norm='batch', activ='relu')


        # 32x32x32 to 3x64x64 (output = 3)
        self.up_conv5 = up_conv(conv_dim, 3, kernel_size=3, norm=None, activ='tanh')

    def forward(self, z):
        """
        Generate an image given a sample of random noise.

        Input
        -----
            z: BS x noise_size x 1 x 1   -->  16x100x1x1

        Output
        ------
            out: BS x channels x image_width x image_height  -->  16x3x64x64
        """
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        layer_out = self.up_conv1(z)
        layer_out = self.up_conv2(layer_out)
        layer_out = self.up_conv3(layer_out)
        layer_out = self.up_conv4(layer_out)
        layer_out = self.up_conv5(layer_out)

        return layer_out


class ResnetBlock(nn.Module):

    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1, norm=norm,
            activ=activ
        )

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out



class DCDiscriminator(nn.Module):
    """Architecture of the discriminator network."""

    def __init__(self, conv_dim=64, norm='instance'):
        super().__init__()
        # 3x64x64 -> 32x32x32 
        self.conv1 = conv(3, 32, 4, 2, 1, norm, False, 'relu')

        # 32x32x32 -> 64x16x16
        self.conv2 = conv(32, 64, 4, 2, 1, norm, False, 'relu')

        # 64x16x16 -> 128x8x8
        self.conv3 = conv(64, 128, 4, 2, 1, norm, False, 'relu')

        # 128x8x8 -> 256x4x4
        self.conv4 = conv(128, 256, 4, 2, 1, norm, False, 'relu')

        # need a single image after this layer.
        # 256x4x4 -> 1x1x1
        self.conv5 = conv(256, 1, 4, 2, 0, None, False, None)

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()
    
####
# Spectral Norm Function 

# define spectral conv layer
class SpectralNormConv(nn.Module):
    """Create a convolutional layer, with optional normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, bias=True):
        super().__init__()

        # from MyConv2D

        # define weight
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        # shape of bias: [out_channels]
        if (bias == True):
            self.b = nn.Parameter(torch.randn(out_channels))
        else:    
            self.b = None

        # power iteration step from paper.  
        #  v_l ← (W^l)^Tu l/||(W^l)^Tu _l||_2
        #  u _l ← W^lv l/||W^lv _l||_2        
        self.v = torch.randn(in_channels * kernel_size * kernel_size)
        self.v = self.norm(self.v)

        self.u = torch.randn(out_channels)
        self.u = self.norm(self.u)

        # update for the rest 
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    # clac norm
    def norm(self, b):
        norm_out = b / torch.norm(b)
        return norm_out
    
    # every forward: need to normalize W^Tu -> v and normalize Wv -> u and make W_sn = W / (u^T W v)
    def forward(self, z):

        # reshape to 2d matrix 
        W_2D = self.W.view(self.W.size(0), -1)

        # read u, v to GPU
        u_l_1 = self.u.to(self.W.device)
        v_l_1 = self.v.to(self.W.device)

        # power iteration step from paper.  
        #  v_l ← (W^l)^Tu l/||(W^l)^Tu _l||_2
        #  u_l ← W^lv l/||W^lv _l||_2        
        v_l = self.norm(torch.mv(W_2D.t(), u_l_1)).detach()
        u_l = self.norm(torch.mv(W_2D, v_l_1)).detach()

        # update u and v 
        self.u = u_l
        self.v = v_l

        W_spectral_norm = self.W / torch.dot(self.u, torch.mv(W_2D, self.v))
        # W_spectral_norm = W_spectral_norm.view(self.W.size())

        # do th econv2d with the spectral norm W
        out = nn.functional.conv2d(z, W_spectral_norm, bias=self.b, stride=self.stride, padding=self.padding)
        
        return out

# define spectral_norm_conv func similar to conv function
def spectral_norm_conv(in_channels, out_channels, kernel_size, stride=2, padding=1,activ=None):

    layers = []

    conv_layer = SpectralNormConv(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=True
    )
    layers.append(conv_layer)

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

# define spectral norm discriminator
class SpectralNormDiscriminator(nn.Module):
    """Architecture of the Spectral Normdiscriminator network."""

    def __init__(self, conv_dim=64):
        super().__init__()

        # conv parameters 

        # 3x64x64 -> 32x32x32 
        self.conv1 = spectral_norm_conv(3, 32, 4, 2, 1, activ='relu')

        # 32x32x32 -> 64x16x16
        self.conv2 = spectral_norm_conv(32, 64, 4, 2, 1, activ='relu')

        # 64x16x16 -> 128x8x8
        self.conv3 = spectral_norm_conv(64, 128, 4, 2, 1, activ='relu')

        # 128x8x8 -> 256x4x4
        self.conv4 = spectral_norm_conv(128, 256, 4, 2, 1, activ='relu')

        # need a single image after this layer.
        # 256x4x4 -> 1x1x1
        self.conv5 = spectral_norm_conv(256, 1, 4, 2, 0)

    def forward(self, x):
        """Forward pass, x is (B, C, H, W)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()
