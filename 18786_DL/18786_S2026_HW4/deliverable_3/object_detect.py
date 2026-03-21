import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# DELIVERABLE 3:
class Baseline(nn.Module):
    '''
    Docstring for Baseline
    '''

def run_baseline(input_image, image_name, device):

    # INPUT PROCESSING 

    # fetch & convert input 
    input_image = Image.open(input_image)
    # input_image = model_input.convert("RGB")

    # split image into 5 by 5 non-overlapping patches
    image_width = input_image.image_size[0]
    image_height = input_image.image_size[1]

    patch_width = image_width // 5 
    patch_height = image_height // 5
    patch_coord = []
    subimages = [] # array w coordinates of each sub-images
    for i in range(5):
        for j in range(5):
            patch_vals = (patch_width * i, patch_height * j, patch_width * j + patch_width, patch_height * i + patch_height)
            patch_coord.append(patch_vals)
            subimages.append(input_image.crop(patch_vals))

    # set up model: https://docs.pytorch.org/vision/stable/models.html 
    baseline_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    baseline_model.eval() # set model to eval mode
    baseline_model.to(device) # for GPU

    # iterate through all images, classify objects
    # https://pytorch.org/hub/pytorch_vision_resnet/
    preprocess_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess_transform(input_image)
    input_batch = input_tensor.unsqeeze(0)
    input_batch = input_batch.to(device)
    dog_cat_found = []
    curr_index = 0
    for image_i in subimages:
        # classify with the model
        with torch.no_grad(): 
            model_output = baseline_model(image_i)
            probabilities = F.softmax(model_output[0], dim=0)
            score, prediction = torch.max(probabilities, 1)
            score = score.item()
            prediction = prediction.item()

        # validate detected cat or dog
        if (score > 0.75): # hardcoded confidence threshold to 0.75
            dog_cat_found.append({"image": image_i, "score": score, "prediction": prediction, "subimage_location": patch_coord[curr_index]})
        
        curr_index += 1
    
    # create bounding box around identified cat and dog
    bbox_creation = ImageDraw.Draw(input_image)
    for found_object in dog_cat_found: 

        x1, y1, x2, y2 = dog_cat_found["subimage_location"]
        bbox_creation.rectangle([x1, y1, x2, y2])
        label_location = (x1 + 5, y1 + 5)
        bbox_creation.text(label_location[0], label_location[1], f"{prediction}")

        input_image.save(f"{image_name}_baseline_image")


if __name__ == "__main__":

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'

    run_baseline("..\2007_001239.jpg", "2007_image", device)