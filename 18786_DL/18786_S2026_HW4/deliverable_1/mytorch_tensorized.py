import sys
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from google.colab import drive

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

# CONV2D CLASS IMPLEMENTATION 
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
        self.W = nn.Parameter(0.01 * torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        # shape of bias: [out_channels]
        if (bias == True):
            self.b = nn.Parameter(torch.zeros(out_channels))
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

        # CHECK: (left, right, top, bottom) = (self.padding, self.padding, self.padding, self.padding)
        # input_w_padding = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="constant", value=0)

        output_height = ((input_height + 2 * self.padding - (self.kernel_size -1)- 1) // self.stride + 1)
        output_width = ((input_width + 2 * self.padding - (self.kernel_size -1) -1) // self.stride + 1)
        
        # # CHECK: Create a tensor of needed size
        # output = torch.zeros(batch_size, self.out_channels, output_height, output_width, device=x.device, dtype=x.dtype)

        # SLIDING WINDOW: can extract from tensor using unfold: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
        # unfold 1: (batch size, in channels, out channel, w_pad, K)
        # unfold 2: (batch size, in channels, out channel, w_out, K, K)
        input_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride) # get the windows in the height dim

        # reshape
        w_flat = self.W.view(self.out_channels, -1)

        input_unfold = input_unfold.transpose(1, 2)

        # do convolution 
        output = torch.matmul(input_unfold, w_flat.t())

        # bias addition
        if self.b != None: 
            output += self.b.view(1,1,self.out_channels)

        output = output.view(batch_size, output_height, output_width, self.out_channels)

        # standard conv output 
        output = output.permute(0,3,1,2).contiguous()

        return output
        # raise NotImplementedError

# MAXPOOL2D CLASS IMPLEMENTATION 
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

        ## Maxpooling process
        ## Feel free to use for loop
        # ----- TODO -----
        input_unfold = x.unfold(2, self.kernel_size, self.stride) # get the windows in the height dim
        input_unfold = input_unfold.unfold(3, self.kernel_size, self.stride)

        output = input_unfold.amax(dim=(-1,-2))

        return output
    
# FCNN CLASS IMPLEMENTATION 
class MyFCNN(nn.Module):
    def __init__(self, hidden_layers_size, input_size=None, num_output_classes=None):

        """
        My custom FCNN, designed for CIFAR-100. 

        CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html -> 32x32 colour images, 3 channels (read, green, blue) (1024 bytes each), 100 classes

        [input]
        * input_size  : dimensions of input (flat input)
        * hidden_layers_size  : list of hidden layer sizes # final model: 2 hidden layers [satisfies > 1 hidden layer requirement]
        * num_output_classes   : number of classes for output

        """
        super().__init__()

        self.input_size = input_size
        if (input_size == None):
            # CIFAR Image data: 32 x 32 images, 3 color channels
            self.input_size = 32 * 32 * 3
        self.num_output_classes = num_output_classes
        if (num_output_classes == None):
            self.num_output_classes = 100

        self.hidden_layers_size = hidden_layers_size

        # hidden layer 1 on NN. 
        self.hidden_layer_1 = nn.Linear(self.input_size, self.hidden_layers_size[0])
        self.layer_1_activation = nn.ReLU()

        # hidden layer 2 
        self.hidden_layer_2 = nn.Linear(self.hidden_layers_size[0], self.hidden_layers_size[1])
        self.layer_2_activation = nn.ReLU()

        # output layer 
        self.output_layer = nn.Linear(self.hidden_layers_size[1], self.num_output_classes)

        # for flatten of input
        self.flat = nn.Flatten()
    
    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):

        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        """

        # flatten input
        x = self.flat(x)

        # hidden layer 1 
        x = self.hidden_layer_1(x)
        x = self.layer_1_activation(x)

        # # hidden layer 2
        x = self.hidden_layer_2(x)
        x = self.layer_2_activation(x)

        # output layer
        x = self.output_layer(x)

        return x

# CNN CLASS IMPLEMENTATION
class MyCNN(nn.Module):
    def __init__(self, num_output_classes=None):

        """
        My custom CNN, designed for CIFAR-100. 

        Pattern: 
        - Conv
        - Activation (ReLU)
        - Max Pooling

        [input]
        * num_output_classes   : number of classes for output
        """
        super().__init__()

        self.num_output_classes = num_output_classes
        if (num_output_classes == None):
            self.num_output_classes = 100        

        # LAYER 1
        self.layer_1_conv = MyConv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer_1_relu = nn.ReLU()
        self.layer_1_maxpool = MyMaxPool2D(kernel_size=2, stride=2)

        # LAYER 2
        self.layer_2_conv = MyConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer_2_relu = nn.ReLU()
        self.layer_2_maxpool = MyMaxPool2D(kernel_size=2, stride=2)

        # LAYER 3
        # self.layer_3_conv = MyConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        # self.layer_3_relu = nn.ReLU()
        # self.layer_3_maxpool = MyMaxPool2D(kernel_size=2, stride=2)

        # for flatten before fc layer
        self.flat = nn.Flatten()

        # dropout layer
        self.dropout = nn.Dropout(p=0.3)

        # OUTPUT LAYER
        self.fully_connected_hidden_layer = nn.Linear(4096, 256) # flattened input to first FC layer. 
        self.fully_connected_relu = nn.ReLU()
        self.fully_connected_layer_out = nn.Linear(256, self.num_output_classes)

    
    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):

        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        """
        # print("TEST: forward pass")

        # Hidden layer structure: conv -> relu -> pool
        # hidden layer 1 
        x = self.layer_1_conv(x)
        x = self.layer_1_relu(x)
        x = self.layer_1_maxpool(x)

        # hidden layer 2 
        x = self.layer_2_conv(x)
        x = self.layer_2_relu(x)
        x = self.layer_2_maxpool(x)

        # hidden layer 3 
        # x = self.layer_3_conv(x)
        # x = self.layer_3_relu(x)
        # x = self.layer_3_maxpool(x)

        # Flatten before you apply fully connected layers
        x = self.flat(x)

        # FC layer to calc class scores. 
        x = self.fully_connected_hidden_layer(x)
        x = self.fully_connected_relu(x)
        x = self.fully_connected_layer_out(x)

        return x

# Train the Model: 
# -> https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def train(net, num_epoch, learning_rate, train_dataloader, test_dataloader, device):
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_loss = []
    train_accuracy = []
    
    test_loss = []
    test_accuracy = []

    for epoch in range(num_epoch):

        # print("starting batch")

        # CHECK: FIND EVIDENCE
        net.train()

        curr_loss = 0.0
        curr_correct = 0 
        curr_total = 0

        for i, data in enumerate(train_dataloader,0):
            inputs, labels = data

            # if CUDA
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero param gradients
            optimizer.zero_grad()

            # forward & backward & optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # output stats
            curr_loss += inputs.size(0) * loss.item() # mult by num images

            # class w highest val = prediction
            _, prediction = torch.max(outputs, 1)

            # update tracking variables for output
            curr_total += labels.size(0)
            curr_correct += (prediction == labels).sum().item()

        # calculate & update loss & accuracy 
        train_loss_i = curr_loss / curr_total
        train_loss.append(train_loss_i)

        train_accuracy_i = curr_correct / curr_total
        train_accuracy.append(train_accuracy_i)

        # CHECK: FIND EVIDENCE
        net.eval()

        # Evaluation Section
        test_correct = 0
        test_total = 0
        curr_test_loss = 0.0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data

                # to GPU
                images = images.to(device)
                labels = labels.to(device)

                # calc ouputs through network                
                outputs = net(images)

                # calc loss for calculations
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                curr_test_loss += labels.size(0) * loss.item()
        
        test_loss_i = curr_test_loss / test_total
        test_loss.append(test_loss_i)

        test_accuracy_i = test_correct / test_total
        test_accuracy.append(test_accuracy_i)

        # CHECK: REMOVE -> ONLY FOR TRACKING / TRAINING
        print(
            f"Epoch [{epoch+1}/{num_epoch}] | "
            f"Train Loss: {train_loss_i:.4f} | Train Acc: {train_accuracy_i:.4f} | "
            f"Test Loss: {test_loss_i:.4f} | Test Acc: {test_accuracy_i:.4f}"
        )

    return net, train_loss, train_accuracy, test_loss, test_accuracy
    
def generate_plots(model_str, epochs, train_loss, train_accuracy, test_loss, test_accuracy):

    # make the losses plot 
    plt.figure()

    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")

    plt.title(f"{model_str} Train and Test Loss Plot")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"{model_str}_loss_plot")

    # make the accuracy plot
    plt.figure()

    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, test_accuracy, label="Test Accuracy")

    plt.title(f"{model_str} Train and Test Accuracy Plot")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"{model_str}_accuracy_plot")

# Visualization: https://medium.com/@vikrampande783/visualizing-convolutional-networks-787810e1f6ea
def visualize_preds(model, model_str, dataset, classes, device):

    # added to disable dropout & batchnrm.
    model.eval()

    # pick a random selection of 5 images to view
    selected_images = [5, 10, 95, 331, 789]

    # define figure
    fig = plt.figure(figsize=(30, 10))

    image_pos = 0
    with torch.no_grad():
        for image_i in selected_images:
            input, label = dataset[image_i]

            input_image = input.unsqueeze(0) # -> (batch size, channels, height, width)
            input_image = input_image.to(device) # move to gpu

            output = model(input_image)
            _, prediction = torch.max(output, 1)
            prediction = prediction.item() # -> make tensor into int

            plt.subplot(1, 5, image_pos + 1) # create subplot w axes for current iamge
            plt.imshow(to_pil_image(input))
            plt.title(f"Ground Truth: {classes[label]} | Pred: {classes[prediction]}", fontsize=18)

            plt.axis("off") # ADDED to remove tick marks

            # update image position counter 
            image_pos += 1
    
    plt.tight_layout() # Added for subplot adjusting
    plt.savefig(f"{model_str}_predictions", dpi=300, bbox_inches="tight")

    plt.close()

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
        print("Running Conv Test 3.")
        torch.manual_seed(0)

        batch_size = 2
        in_channels = 6 # applies convolution on each in_channel seperately & take sum
        out_channels = 7 # make _ feature maps
        height = 16
        width = 16
        padding = 1 
        stride = 3
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
        print("Running Conv Test 4.")
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
        print("Running Conv Test 5.")
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
        print("Running MaxPool Test 2.")
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
        print("Running MaxPool Test 3.")
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

    # RUN ALL CONV TESTS
    conv_test_1()
    conv_test_2()
    conv_test_3()
    conv_test_4()
    conv_test_5()

    # RUN ALL MAXPOOL TESTS
    maxpool_test_1()
    maxpool_test_2()
    maxpool_test_3()

    # MODEL TRAINING FOR DELIVERABLE 1

    # SET TRAINING PARAM
    batch_size = 64
    epochs = 10
    learning_rate = 0.01

    # drive.mount('/content/drive')
    data_root = "./data"

    # TODO check: 
    # init data loaders & data: https://docs.pytorch.org/vision/0.9/datasets.html#cifar
    train_data = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'
    
    # FCNN
    # my_fcnn = MyFCNN(hidden_layers_size=[512, 256], num_output_classes=100)
    # my_fcnn, train_loss, train_accuracy, test_loss, test_accuracy = train(net=my_fcnn, num_epoch=epochs, learning_rate=learning_rate, train_dataloader=train_dataloader, test_dataloader=test_dataloader, device=device)

    # # PLOTTING
    # num_epochs = len(train_loss)
    # epochs_axis = [i for i in range(1, num_epochs + 1)]
    # generate_plots("FCNN", epochs_axis, train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)
    # visualize_preds(my_fcnn, "FCNN", test_data, train_data.classes, device)

    # CNN 
    my_cnn = MyCNN(num_output_classes=100)
    cnn, train_loss, train_accuracy, test_loss, test_accuracy = train(net=my_cnn, num_epoch=epochs, learning_rate=learning_rate, train_dataloader=train_dataloader, test_dataloader=test_dataloader, device=device)

    # PLOTTING
    num_epochs = len(train_loss)
    epochs_axis = [i for i in range(1, num_epochs + 1)]
    generate_plots("CNN", epochs_axis, train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)
    visualize_preds(my_cnn, "CNN", test_data, train_data.classes, device)


