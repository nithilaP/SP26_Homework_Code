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
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

# DELIVERABLE 3:
class Baseline(nn.Module):
    '''
    Docstring for Baseline
    '''

def run_baseline(input_image):

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'

    # set up model: https://docs.pytorch.org/vision/stable/models.html 
    baseline_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval() # set model to eval mode
    baseline_model.to(device) # for GPU 


if __name__ == "__main__":
