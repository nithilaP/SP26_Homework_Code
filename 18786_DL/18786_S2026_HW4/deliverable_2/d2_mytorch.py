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

        print("starting batch")

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


            image = input.cpu().permute(1,2,0).numpy()
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
    batch_size = 64
    epochs = 10
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 0
    scheduler_step_size = 10
    scheduler_gamma = 0.1

    # data transformations: experiment with augmentations here (random crop, etc)
    train_transformation = transforms.Compose([transforms.ToTensor()])
    test_transformation = transforms.Compose([transforms.ToTensor()])

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
    
    # CNN 
    my_cnn = MyCNN(num_output_classes=100)
    cnn, train_loss, train_accuracy, test_loss, test_accuracy = train(net=my_cnn, num_epoch=epochs, learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay, 
                                                                      scheduler_step_size=scheduler_step_size, scheduler_gamma=scheduler_gamma,train_dataloader=train_dataloader, 
                                                                      test_dataloader=test_dataloader, device=device)

    # PLOTTING
    num_epochs = len(train_loss)
    epochs_axis = [i for i in range(1, num_epochs + 1)]
    generate_plots("CNN", epochs_axis, train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)
    visualize_preds(my_cnn, "CNN", test_data, train_data.classes, device)

