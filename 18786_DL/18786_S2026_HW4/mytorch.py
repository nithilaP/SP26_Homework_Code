import sys
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

# FCNN CLASS IMPLEMENTATION 
class MyFCNN(nn.Module):
    def __init__(self, hidden_layers_size, input_size=None, num_output_classes=None):

        """
        My custom FCNN, designed for CIFAR-100. 

        CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html -> 32x32 colour images, 3 channels (read, green, blue) (1024 bytes each), 100 classes

        [input]
        * input_size  : dimensions of input (flat input)
        * hidden_layers_size  : list of hidden layer sizes # assume 2 hidden layers [satisfies > 1 hidden layer requirement]
        * num_output_classes   : number of classes for output

        """
        super().__init__()

        self.input_size = input_size
        if (input_size == None):
            self.input_size = 32 * 32 * 3 # CIFAR Image data: 32 x 32 images, 3 color channels

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
        self.output_layer =  nn.Linear(self.hidden_layers_size[1], self.num_output_classes)
    
        # layer_activations = []
        # layer_input_size = self.input_size
        # for i in range(num_hidden_layers):
        #     layer_activations.append(nn.Linear(layer_input_size, self.hidden_layers_size[i]))
        #     layer_activations.append(nn.ReLU())
        #     layer_input_size = self.hidden_layers_size[i] #update for next iteration
        # layer_activations.append(nn.Linear(layer_input_size, self.num_output_classes)) # last layer w output


    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):

        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        """

        # turn into 2D tensor: nn.Flatten()
        flat = nn.Flatten()
        x = flat(x)

        # hidden layer 1 
        x = self.hidden_layer_1(x)
        x = self.layer_1_activation(x)

        # hidden layer 2
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
        self.layer_1_activation = nn.ReLU()
        self.layer_1_maxpool = MyMaxPool2D(kernel_size=2, stride=2)

        # LAYER 2
        self.layer_2_conv = MyConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer_2_activation = nn.ReLU()
        self.layer_2_maxpool = MyMaxPool2D(kernel_size=2, stride=2)

        # LAYER 3
        self.layer_3_conv = MyConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer_3_activation = nn.ReLU()
        self.layer_3_maxpool = MyMaxPool2D(kernel_size=2, stride=2)

        # OUTPUT LAYER
        self.fc_1 = nn.Linear(128 * 4 * 4, 256) # assuming hidden layer dim
        self.output_layer_activation = nn.ReLU()
        self.output_layer_func = nn.Linear(256, self.num_output_classes)
    
    def __call__(self, x):
        
        return self.forward(x)
    
    def forward(self, x):

        """
        [input]
        x (torch.tensor)      : (batch_size, in_channels, input_height, input_width)

        [output]
        output (torch.tensor) : (batch_size, out_channels, output_height, output_width)

        """
        # FIX: turn into 2D tensor: nn.Flatten() -> https://stackoverflow.com/questions/65993494/difference-between-torch-flatten-and-nn-flatten
        flat = nn.Flatten()

        # hidden layer 1 
        x = self.layer_1_conv(x)
        x = self.layer_1_activation(x)
        x = self.layer_1_maxpool(x)

        # hidden layer 2 
        x = self.layer_2_conv(x)
        x = self.layer_2_activation(x)
        x = self.layer_2_maxpool(x)

        # hidden layer 3 
        x = self.layer_3_conv(x)
        x = self.layer_3_activation(x)
        x = self.layer_3_maxpool(x)

        # FIX: flatten
        x = flat(x)

        # output layer
        x = self.fc_1(x)
        x = self.output_layer_activation(x)
        x = self.output_layer_func(x)

        return x

# TODO CHECK: model training
def train(net, num_epoch, learning_rate, train_dataloader, test_dataloader, device):
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_loss = []
    train_accuracy = []
    
    test_loss = []
    test_accuracy = []
    # https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html -> Train the network section
    for epoch in range(num_epoch):
        net.train() # TODO FIND EVIDENCE 

        curr_loss = 0.0
        curr_correct = 0 
        curr_total = 0

        for i, data in enumerate(train_dataloader,0):
            inputs, labels = data
            inputs = inputs.to(device) # for CUDA
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # output stats
            curr_loss += inputs.size(0) * loss.item() # mult by num images
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {curr_loss / 2000:.3f}')

            # TODO CHECK
            _, prediction = torch.max(outputs, 1) # from pytorch documentation -> "Test the network on test data"
            curr_total += labels.size(0)
            curr_correct += (prediction == labels).sum().item()

        # calculate & update loss & accuracy 
        train_loss_i = curr_loss / curr_total
        train_loss.append(train_loss_i)

        train_accuracy_i = curr_correct / curr_total
        train_accuracy.append(train_accuracy_i)

        # TEST TODO CHECK: EVALUATE to get test_loss, test_acc
        net.eval() # TODO FIND EVIDENCE 

        # https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html -> "Test the network on test data"
        test_correct = 0
        test_total = 0
        curr_test_loss = 0.0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data

                images = images.to(device)
                labels = labels.to(device)
                
                outputs = net(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                curr_test_loss += labels.size(0) * loss.item()
        
        test_loss_i = curr_test_loss / test_total
        test_loss.append(test_loss_i)

        test_accuracy_i = test_correct / test_total
        test_accuracy.append(test_accuracy_i)

        # TODO: REMOVE 
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


def visualize_predictions(model, dataset, classes, device, model_name, num_images=5):
    model.eval()

    indices = random.sample(range(len(dataset)), num_images)

    plt.figure(figsize=(15, 3))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            input_image = image.unsqueeze(0).to(device)

            output = model(input_image)
            pred = torch.argmax(output, 1).item()

            img_np = image.permute(1, 2, 0).cpu().numpy()

            plt.subplot(1, num_images, i + 1)
            plt.imshow(img_np)
            plt.axis("off")
            plt.title(f"GT: {classes[label]}\nPred: {classes[pred]}")

    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_predictions.png")
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

    # TODO check: 
    # init data loaders & data: https://docs.pytorch.org/vision/0.9/datasets.html#cifar
    train_data = datasets.CIFAR100(root="./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.CIFAR100(root="./data", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'
    
    # FCNN
    my_fcnn = MyFCNN(hidden_layers_size=[1024, 512], num_output_classes=100)
    my_fcnn, train_loss, train_accuracy, test_loss, test_accuracy = train(net=my_fcnn, num_epoch=epochs, learning_rate=learning_rate, train_dataloader=train_dataloader, test_dataloader=test_dataloader, device=device)

    # PLOTTING
    num_epochs = len(train_loss)
    epochs_axis = [i for i in range(1, num_epochs + 1)]
    generate_plots("FCNN", epochs_axis, train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)
    visualize_predictions(my_fcnn, test_data, train_data.classes, device, "FCNN", num_images=5)


    # CNN 
    my_cnn = MyCNN(num_output_classes=100)
    cnn, train_loss, train_accuracy, test_loss, test_accuracy = train(net=my_cnn, num_epoch=1, learning_rate=learning_rate, train_dataloader=train_dataloader, test_dataloader=test_dataloader, device=device)

    # PLOTTING
    num_epochs = len(train_loss)
    epochs_axis = [i for i in range(1, num_epochs + 1)]
    generate_plots("CNN", epochs_axis, train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)
    visualize_predictions(my_cnn, test_data, train_data.classes, device, "CNN", num_images=5)


