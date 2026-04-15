# CMU CMU 18-780/6 Homework 4
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

# padding = 3
# Vanilla Basic: Iteration [6500/6500] | D_real_loss: 0.1132 | D_fake_loss: 0.0006 | G_loss: 6.7294
# Advanced: Iteration [6500/6500] | D_real_loss: 0.0032 | D_fake_loss: 0.0835 | G_loss: 5.4473


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
        # output = (input - kernel_size + 2*padding) / stride) + 1
        self.up_conv1 = nn.Sequential(nn.Conv2d(noise_size, conv_dim * 8, kernel_size=4, stride=1, padding=0, bias=False),
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
        print(f"z shape: {z.shape}")
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
