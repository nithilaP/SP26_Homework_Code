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

# DELIVERABLE 2: OPTIMIZED CNN CLASS IMPLEMENTATION
class Baseline(nn.Module):
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

        # d1 baseline: 2 (conv + batch_norm + relu + pool) + flatten + fc + relu + dropout + fc
        # nn.Conv2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # nn.BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # nn.ReLU: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # nn.Dropout: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.conv_layers = nn.Sequential(
            
            # LAYER 1 
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # LAYER 2 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            # for flatten before fc layer
            nn.Flatten(),

            # OUTPUT LAYER
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(256, self.num_output_classes),
        )
        
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

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x

# BasicAlexNET
class Basic_AlexNET(nn.Module):
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

        # d1 baseline: 2 (conv + batch_norm + relu + pool) + flatten + fc + relu + dropout + fc
        # alexnet : 5 Conv w batch_norm, 3 FC 
        # nn.Conv2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # nn.BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # nn.ReLU: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # nn.Dropout: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.conv_layers = nn.Sequential(
            
            # LAYER 1 
            # Conv(3→96) → BN(96) → ReLU → MaxPool
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # LAYER 2 
            # Conv(96→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv(256→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            # for flatten before fc layer
            nn.Flatten(),

            # FC HIDDEN LAYER 1 
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            # FC OUTPUT LAYER 
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(4096, self.num_output_classes),
        )
    
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

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x

# Smaller FC Layer AlexNET
class Smaller_FC_AlexNET(nn.Module):
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

        # d1 baseline: 2 (conv + batch_norm + relu + pool) + flatten + fc + relu + dropout + fc
        # alexnet : 5 Conv w batch_norm, 3 FC 
        # nn.Conv2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # nn.BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # nn.ReLU: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # nn.Dropout: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.conv_layers = nn.Sequential(
            
            # LAYER 1 
            # Conv(3→96) → BN(96) → ReLU → MaxPool
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # LAYER 2 
            # Conv(96→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv(256→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            # for flatten before fc layer
            nn.Flatten(),

            # FC HIDDEN LAYER 1 
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            # FC OUTPUT LAYER 
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, self.num_output_classes),
        )
    
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

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x

# Deeper FC Layer AlexNET
class Deeper_FC_AlexNET(nn.Module):
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

        # d1 baseline: 2 (conv + batch_norm + relu + pool) + flatten + fc + relu + dropout + fc
        # alexnet : 5 Conv w batch_norm, 3 FC 
        # nn.Conv2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # nn.BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # nn.ReLU: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # nn.Dropout: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.conv_layers = nn.Sequential(
            
            # LAYER 1 
            # Conv(3→96) → BN(96) → ReLU → MaxPool
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # LAYER 2 
            # Conv(96→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv(256→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            # for flatten before fc layer
            nn.Flatten(),

            # FC HIDDEN LAYER 1 
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            # FC HIDDEN LAYER 2 
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            # FC HIDDEN LAYER 3 
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            # FC OUTPUT LAYER 
            nn.Linear(512, self.num_output_classes),
        )
    
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

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x

# Deeper Conv Layer AlexNET
class Deeper_Conv_AlexNET(nn.Module):
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

        # d1 baseline: 2 (conv + batch_norm + relu + pool) + flatten + fc + relu + dropout + fc
        # alexnet : 5 Conv w batch_norm, 3 FC 
        # nn.Conv2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # nn.BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # nn.ReLU: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # nn.Dropout: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.conv_layers = nn.Sequential(
            
            # LAYER 1 
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),

            # LAYER 2 
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv(384→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            # for flatten before fc layer
            nn.Flatten(),

            # FC HIDDEN LAYER 1 
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            # FC OUTPUT LAYER 
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, self.num_output_classes),
        )
    
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

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x

# No Dropout with Smaller FC Layer AlexNET
class No_Dropout_Smaller_FC_AlexNET(nn.Module):
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

        # d1 baseline: 2 (conv + batch_norm + relu + pool) + flatten + fc + relu + dropout + fc
        # alexnet : 5 Conv w batch_norm, 3 FC 
        # nn.Conv2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # nn.BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # nn.ReLU: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # nn.Dropout: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.conv_layers = nn.Sequential(
            
            # LAYER 1 
            # Conv(3→96) → BN(96) → ReLU → MaxPool
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # LAYER 2 
            # Conv(96→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv(256→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            # for flatten before fc layer
            nn.Flatten(),

            # FC HIDDEN LAYER 1 
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),

            # FC OUTPUT LAYER 
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_output_classes),
        )
    
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

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x

# Bigger Dropout Layer AlexNET
class Bigger_Dropout_AlexNet(nn.Module):
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

        # d1 baseline: 2 (conv + batch_norm + relu + pool) + flatten + fc + relu + dropout + fc
        # alexnet : 5 Conv w batch_norm, 3 FC 
        # nn.Conv2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # nn.BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # nn.ReLU: https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # nn.Linear: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # nn.Dropout: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.conv_layers = nn.Sequential(
            
            # LAYER 1 
            # Conv(3→96) → BN(96) → ReLU → MaxPool
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # LAYER 2 
            # Conv(96→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv(256→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→384) → BN(384) → ReLU
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),

            # Conv(384→256) → BN(256) → ReLU → MaxPool
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            # for flatten before fc layer
            nn.Flatten(),

            # FC HIDDEN LAYER 1 
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            # FC OUTPUT LAYER 
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.num_output_classes),
        )
    
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

        x = self.conv_layers(x)
        x = self.fc_layers(x)

        return x

# Train the Model: 
# -> https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Learning Rate Scheduler: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html 
def train(net, num_epoch, learning_rate, momentum, weight_decay, scheduler_step_size, scheduler_gamma, train_dataloader, test_dataloader, device):
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    train_loss = []
    train_accuracy = []
    
    test_loss = []
    test_accuracy = []

    for epoch in range(num_epoch):

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

        # step on the scheduler if within the epochs step size
        scheduler.step()

        # CHECK: REMOVE -> ONLY FOR TRACKING / TRAINING
        print(
            f"Epoch [{epoch+1}/{num_epoch}] | "
            f"LR: {scheduler.get_last_lr()[0]:.5f} | "
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

    # pick a random selection of 10 images to view
    selected_images = [5, 10, 95, 129, 267, 331, 490, 671, 789, 890]

    # define figure
    fig = plt.figure(figsize=(20, 10))

    image_pos = 0
    with torch.no_grad():
        for image_i in selected_images:
            input, label = dataset[image_i]

            input_image = input.unsqueeze(0) # -> (batch size, channels, height, width)
            input_image = input_image.to(device) # move to gpu

            output = model(input_image)
            _, prediction = torch.max(output, 1)
            prediction = prediction.item() # -> make tensor into int

            plt.subplot(2, 5, image_pos + 1) # create subplot w axes for current iamge

            # Rearranges it into (height, weight, channel) and to numpy array
            # -> https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
            image = input.cpu().permute(1,2,0).numpy()

            # Clip data: https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa
            image = np.clip(image, 0, 1)
            plt.imshow(image)

            # plt.imshow(to_pil_image(input))

            plt.title(f"Ground Truth: {classes[label]} | Pred: {classes[prediction]}")

            plt.axis("off") # ADDED to remove tick marks

            # update image position counter 
            image_pos += 1
    
    plt.tight_layout() # Added for subplot adjusting

    plt.savefig(f"{model_str}_predictions")
    plt.close()

if __name__ == "__main__":

    # drive.mount('/content/drive')
    data_root = "./data"

    # MODEL TRAINING FOR DELIVERABLE 2

    # TUNABLE PARAM FOR ABLATION STUDY 
    batch_size = 128
    epochs = 50
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    scheduler_step_size = 20
    scheduler_gamma = 0.01

    # data transformations: experiment with augmentations here (random crop, etc)
    # normalization values used for CIFAR-100: 
    #   Normalize accorindg to ImageNet values:  mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
    # -> https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    # -> https://pytorch.org/vision/stable/models.html
    # train_transformation = transforms.Compose([transforms.ToTensor(), 
    #                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    # test_transformation = transforms.Compose([transforms.ToTensor(), 
    #                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    train_transformation = transforms.Compose([transforms.ToTensor(), 
                                               transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))])
    test_transformation = transforms.Compose([transforms.ToTensor(), 
                                               transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))])

    # train_transformation_normalized = transforms.Compose([transforms.ToTensor()])
    # test_transformation = transforms.Compose([transforms.ToTensor()])

    # init data loaders & data: https://docs.pytorch.org/vision/0.9/datasets.html#cifar
    train_data = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transformation)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_transformation)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'
    
    # baseline CNN 
    my_baseline_cnn = Smaller_FC_AlexNET(num_output_classes=100)
    my_baseline_cnn, train_loss, train_accuracy, test_loss, test_accuracy = train(net=my_baseline_cnn, num_epoch=epochs, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay, 
                                                                      scheduler_step_size=scheduler_step_size, scheduler_gamma=scheduler_gamma,train_dataloader=train_dataloader, 
                                                                      test_dataloader=test_dataloader, device=device)

    # PLOTTING
    num_epochs = len(train_loss)
    epochs_axis = [i for i in range(1, num_epochs + 1)]
    generate_plots("CIFAR100_Norm_AlexNet", epochs_axis, train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)
    visualize_preds(my_baseline_cnn, "CIFAR100_Norm_AlexNet", test_data, train_data.classes, device)

