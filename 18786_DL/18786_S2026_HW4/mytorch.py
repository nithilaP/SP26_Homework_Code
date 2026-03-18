import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):

        """
        My custom Convolution 2D layer.

        [input]
        * in_channels  : input channel number
        * out_channels : output channel number
        * kernel_size  : kernel size
        * stride       : stride size
        * padding      : padding size
        * bias         : taking into account the bias term or not (bool)

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        ## Create the torch.nn.Parameter for the weights and bias (if bias=True)
        ## Be careful about the size
        # ----- TODO -----

        # shape of weight: [out_channels, in_channels, kernel_size, kernel_size]
        self.W = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        # shape of bias: [out_channels]
        if (bias == True):
            self.b = nn.Parameter(torch.randn(out_channels))
        else:    
            self.b = None

        # raise NotImplementedError
            
    
    def __call__(self, x):
        
        return self.forward(x)


    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)
        """

        # call MyFConv2D here
        # ----- TODO -----

        batch_size = x.shape[0]
        channel = x.shape[1]
        input_height = x.shape[2]
        input_width = x.shape[3]

        # calculating dims: 
        # dim_out = (dim_in + 2P - D(K-1) - 1) / S + 1 
        # P - Padding, D - Dilation, K - Kernel Size, S - Stride length (dilation=1 in this case)

        # CHECK TO_DO: need padding to make sure spatial dim are preserved of the input after convolution 
        # input_padding = nn.ZeroPad2d(self.padding)

        # TO_DO: (left, right, top, bottom) = (self.padding, self.padding, self.padding, self.padding)
        input_w_padding = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="constant", value=0)

        output_height = ((input_height + 2 * self.padding - (self.kernel_size -1)- 1) // self.stride + 1)
        output_width = ((input_width + 2 * self.padding - (self.kernel_size -1) -1) // self.stride + 1)
        
        # TODO: DOUBLE CHECK BELOW!
        output = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=x.dtype)

        # iterate through each dimension of output tensor
        for batch in range(batch_size): 
            for out_channel in range(self.out_channels): 
                for output_i in range(output_height):
                    for output_j in range(output_width):

                        # determine the kernel window 
                        # TODO: CHECK THIS!! !
                        kernel_window = input_w_padding[batch, :, (output_i * self.stride):(output_i * self.stride + self.kernel_size),  (output_j * self.stride):(output_j * self.stride + self.kernel_size)]

                        # do the convolution 
                        # TODO: CHECK THIS!! !
                        output[batch, out_channel, output_i, output_j] = torch.sum(self.W[out_channel] * kernel_window)
                        if self.b != None: 
                            output[batch, out_channel, output_i, output_j] += self.b[out_channel]
        return output
        # raise NotImplementedError

    
class MyMaxPool2D(nn.Module):

    def __init__(self, kernel_size, stride=None):
        
        """
        My custom MaxPooling 2D layer.
        [input]
        * kernel_size  : kernel size
        * stride       : stride size (default: None)
        """
        super().__init__()
        self.kernel_size = kernel_size

        ## Take care of the stride
        ## Hint: what should be the default stride_size if it is not given? 
        ## Think about the relationship with kernel_size
        # ----- TODO -----
        if (stride == None):
            self.stride = kernel_size
        else:
            self.stride = stride

        # raise NotImplementedError


    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):
        
        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        [hint]
        * out_channel == in_channel
        """
        
        ## check the dimensions
        self.batch_size = x.shape[0]
        self.channel = x.shape[1]
        self.input_height = x.shape[2]
        self.input_width = x.shape[3]
        
        ## Derive the output size
        # ----- TODO -----
        # no padding value for this case. -> dim_out = (dim_in - D(K-1) - 1) / S + 1 

        self.output_height   = ((self.input_height - (self.kernel_size -1) - 1) // self.stride + 1)
        self.output_width    = ((self.input_width - (self.kernel_size -1) - 1) // self.stride + 1)
        self.output_channels = self.channel

        output = torch.zeros(self.batch_size, self.output_channels, self.output_height, self.output_width, device=x.device, dtype=x.dtype)

        ## Maxpooling process
        ## Feel free to use for loop
        # ----- TODO -----

        # iterate through each dimension of output tensor
        for batch in range(self.batch_size): 
            for out_channel in range(self.output_channels): 
                for output_i in range(self.output_height):
                    for output_j in range(self.output_width):

                        # determine the kernel window 
                        # TODO: CHECK THIS!! !
                        kernel_window = x[batch, out_channel, (output_i * self.stride):(output_i * self.stride + self.kernel_size),  (output_j * self.stride):(output_j * self.stride + self.kernel_size)]

                        # do the convolution 
                        # TODO: CHECK THIS!! !
                        output[batch, out_channel, output_i, output_j] = torch.max(kernel_window)

        return output

        # raise NotImplementedError


# BUILD FCNN FOR CIFAR-100 


if __name__ == "__main__":

    ## Test your implementation!

    # CONV TEST 1
    def conv_test_1():
        print("Running Conv Test 1.")
        torch.manual_seed(0)

        batch_size = 2
        in_channels = 3 # applies convolution on each in_channel seperately & take sum
        out_channels = 4 # make _ feature maps
        height = 16
        width = 16
        padding = 1 
        stride = 1
        kernel_size = 3

        x = torch.randn(batch_size, in_channels, height, width)

        my_conv = MyConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        out_my_conv = my_conv(x)
        print("my_conv output: ", out_my_conv.shape)

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        # need to copy the weight into torch conv
        # TODO: CHECK IN 
        conv.weight.data = my_conv.W.data.clone()
        conv.bias.data = my_conv.b.data.clone()
        out_conv = conv(x)
        print("conv output: ", out_conv.shape)

        # check if the shapes match 
        shape_check = (out_my_conv.shape == out_conv.shape)
        print("Compare Shapes: ", shape_check)

        # compare the outputs 
        # TODO: CHECK
        compare_output = torch.allclose(out_my_conv, out_conv, atol=1e-5)
        print("Compare Output: ", compare_output)
    
    # CONV TEST 2
    def conv_test_2():
        print("Running Conv Test 2.")
        torch.manual_seed(0)

        batch_size = 2
        in_channels = 6 # applies convolution on each in_channel seperately & take sum
        out_channels = 7 # make _ feature maps
        height = 16
        width = 16
        padding = 1 
        stride = 1
        kernel_size = 5

        x = torch.randn(batch_size, in_channels, height, width)

        my_conv = MyConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        out_my_conv = my_conv(x)
        print("my_conv output: ", out_my_conv.shape)

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        # need to copy the weight into torch conv
        # TODO: CHECK IN 
        conv.weight.data = my_conv.W.data.clone()
        conv.bias.data = my_conv.b.data.clone()
        out_conv = conv(x)
        print("conv output: ", out_conv.shape)

        # check if the shapes match 
        shape_check = (out_my_conv.shape == out_conv.shape)
        print("Compare Shapes: ", shape_check)

        # compare the outputs 
        # TODO: CHECK
        compare_output = torch.allclose(out_my_conv, out_conv, atol=1e-5)
        print("Compare Output: ", compare_output)
    
        # CONV TEST 2
    
    # CONV TEST 3: stride > 1 
    def conv_test_3():
        print("Running Conv Test 2.")
        torch.manual_seed(0)

        batch_size = 2
        in_channels = 6 # applies convolution on each in_channel seperately & take sum
        out_channels = 7 # make _ feature maps
        height = 16
        width = 16
        padding = 1 
        stride = 1
        kernel_size = 5

        x = torch.randn(batch_size, in_channels, height, width)

        my_conv = MyConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        out_my_conv = my_conv(x)
        print("my_conv output: ", out_my_conv.shape)

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        # need to copy the weight into torch conv
        # TODO: CHECK IN 
        conv.weight.data = my_conv.W.data.clone()
        conv.bias.data = my_conv.b.data.clone()
        out_conv = conv(x)
        print("conv output: ", out_conv.shape)

        # check if the shapes match 
        shape_check = (out_my_conv.shape == out_conv.shape)
        print("Compare Shapes: ", shape_check)

        # compare the outputs 
        # TODO: CHECK
        compare_output = torch.allclose(out_my_conv, out_conv, atol=1e-5)
        print("Compare Output: ", compare_output)
    
    # CONV TEST 4: padding = 0
    def conv_test_4():
        print("Running Conv Test 2.")
        torch.manual_seed(0)

        batch_size = 2
        in_channels = 6 # applies convolution on each in_channel seperately & take sum
        out_channels = 7 # make _ feature maps
        height = 16
        width = 16
        padding = 0 
        stride = 1
        kernel_size = 5

        x = torch.randn(batch_size, in_channels, height, width)

        my_conv = MyConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        out_my_conv = my_conv(x)
        print("my_conv output: ", out_my_conv.shape)

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        # need to copy the weight into torch conv
        # TODO: CHECK IN 
        conv.weight.data = my_conv.W.data.clone()
        conv.bias.data = my_conv.b.data.clone()
        out_conv = conv(x)
        print("conv output: ", out_conv.shape)

        # check if the shapes match 
        shape_check = (out_my_conv.shape == out_conv.shape)
        print("Compare Shapes: ", shape_check)

        # compare the outputs 
        # TODO: CHECK
        compare_output = torch.allclose(out_my_conv, out_conv, atol=1e-5)
        print("Compare Output: ", compare_output)

    # CONV TEST 5: bias = False
    def conv_test_5():
        print("Running Conv Test 2.")
        torch.manual_seed(0)

        batch_size = 2
        in_channels = 6 # applies convolution on each in_channel seperately & take sum
        out_channels = 7 # make _ feature maps
        height = 16
        width = 16
        padding = 1
        stride = 1
        kernel_size = 5
        bias = False

        x = torch.randn(batch_size, in_channels, height, width)

        my_conv = MyConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        out_my_conv = my_conv(x)
        print("my_conv output: ", out_my_conv.shape)

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        # need to copy the weight into torch conv
        # TODO: CHECK IN 
        conv.weight.data = my_conv.W.data.clone()
        if (bias):
            conv.bias.data = my_conv.b.data.clone()
        out_conv = conv(x)
        print("conv output: ", out_conv.shape)

        # check if the shapes match 
        shape_check = (out_my_conv.shape == out_conv.shape)
        print("Compare Shapes: ", shape_check)

        # compare the outputs 
        # TODO: CHECK
        compare_output = torch.allclose(out_my_conv, out_conv, atol=1e-5)
        print("Compare Output: ", compare_output)

    # TESTING CASES FOR CONV: bias = False for conv, stride > 1, no padding
    
    # MAXPOOL TEST 1
    def maxpool_test_1():
        print("Running MaxPool Test 1.")
        torch.manual_seed(0)

        batch_size = 2
        channels = 3
        height = 32
        width = 32
        stride = 2
        kernel_size = 2

        x = torch.randn(batch_size, channels, height, width)

        my_maxpool = MyMaxPool2D(kernel_size=kernel_size, stride=stride)
        out_my_maxpool = my_maxpool(x)
        print("my_maxpool output: ", out_my_maxpool.shape)

        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        # need to copy the weight into torch conv
        # TODO: CHECK IN 
        out_maxpool = maxpool(x)
        print("conv output: ", out_maxpool.shape)

        # check if the shapes match 
        shape_check = (out_my_maxpool.shape == out_maxpool.shape)
        print("Compare Shapes: ", shape_check)

        # compare the outputs 
        # TODO: CHECK
        compare_output = torch.allclose(out_my_maxpool, out_maxpool, atol=1e-5)
        print("Compare Output: ", compare_output)
    
    # MAXPOOL TEST 2: stride != kernel-size
    def maxpool_test_2():
        print("Running MaxPool Test 1.")
        torch.manual_seed(1)

        batch_size = 2
        channels = 8
        height = 12
        width = 12
        stride = 2
        kernel_size = 4

        x = torch.randn(batch_size, channels, height, width)

        my_maxpool = MyMaxPool2D(kernel_size=kernel_size, stride=stride)
        out_my_maxpool = my_maxpool(x)
        print("my_maxpool output: ", out_my_maxpool.shape)

        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        # need to copy the weight into torch conv
        # TODO: CHECK IN 
        out_maxpool = maxpool(x)
        print("conv output: ", out_maxpool.shape)

        # check if the shapes match 
        shape_check = (out_my_maxpool.shape == out_maxpool.shape)
        print("Compare Shapes: ", shape_check)

        # compare the outputs 
        # TODO: CHECK
        compare_output = torch.allclose(out_my_maxpool, out_maxpool, atol=1e-5)
        print("Compare Output: ", compare_output)
    
    # MAXPOOL TEST 3: stride = None
    def maxpool_test_3(): 
        print("Running MaxPool Test 1.")
        torch.manual_seed(1)

        batch_size = 2
        channels = 8
        height = 12
        width = 12
        stride = None
        kernel_size = 4

        x = torch.randn(batch_size, channels, height, width)

        my_maxpool = MyMaxPool2D(kernel_size=kernel_size, stride=stride)
        out_my_maxpool = my_maxpool(x)
        print("my_maxpool output: ", out_my_maxpool.shape)

        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        # need to copy the weight into torch conv
        # TODO: CHECK IN 
        out_maxpool = maxpool(x)
        print("conv output: ", out_maxpool.shape)

        # check if the shapes match 
        shape_check = (out_my_maxpool.shape == out_maxpool.shape)
        print("Compare Shapes: ", shape_check)

        # compare the outputs 
        # TODO: CHECK
        compare_output = torch.allclose(out_my_maxpool, out_maxpool, atol=1e-5)
        print("Compare Output: ", compare_output)

    # RUN ALL TESTS
    conv_test_1()
    conv_test_2()
    conv_test_3()
    conv_test_4()
    conv_test_5()

    maxpool_test_1()
    maxpool_test_2()
    maxpool_test_3()