import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# DELIVERABLE 3: baseline is to split image into 5 patches & classify
def run_baseline(input_image, image_name, device):

    # INPUT PROCESSING 

    # fetch & convert input 
    input_image = Image.open(input_image)
    # input_image = model_input.convert("RGB")

    # split image into 5 by 5 non-overlapping patches
    image_width = input_image.size[0]
    image_height = input_image.size[1]

    patch_width = image_width // 5 
    patch_height = image_height // 5
    patch_coord = []
    subimages = [] # array w coordinates of each sub-images
    for i in range(5): # i: top -> bottom 
        for j in range(5): # j: left -> right
            patch_vals = (patch_width * j, patch_height * i, patch_width * j + patch_width, patch_height * i + patch_height)
            patch_coord.append(patch_vals)
            subimages.append(input_image.crop(patch_vals))

    # set up model: https://docs.pytorch.org/vision/stable/models.html 
    baseline_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    baseline_model.eval() # set model to eval mode
    baseline_model.to(device) # for GPU

    # iterate through all images, classify objects
    # https://pytorch.org/hub/pytorch_vision_resnet/
    # All pretrained model expect input imges to be atleast H =224 x W=224:
    #   https://pytorch.org/hub/pytorch_vision_resnet/#:~:text=All%20pre%2Dtrained%20models%20expect,0.229%2C%200.224%2C%200.225%5D%20.
    preprocess_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dog_cat_found = []
    curr_index = 0
    for image_i in subimages:
        # classify with the model
        with torch.no_grad(): 
            
            # preprocess the image
            input_tensor = preprocess_transform(image_i)
            input_batch = input_tensor.unsqueeze(0)
            input_batch = input_batch.to(device)

            # put image through model
            model_output = baseline_model(input_batch)

            # model_output = unnormalized scores -> run softmas to get probabilities
            probabilities = F.softmax(model_output[0], dim=0)

            # get index with highest score / energy -> choose as prediction
            score, prediction = torch.max(probabilities, dim=0)
            score = score.item() # 
            prediction = prediction.item()

        # validate detected cat or dog
        # ImageNet Classes: https://github.com/pytorch/hub/blob/master/imagenet_classes.txt (for torch pretrained models -> ResNet18)
        # Dog: 152 (Chihuahua) to 269 (Mexican hairless)
        # Cat: 283 (tiger cat) to 294 (cheetah)
        if (score > 0.3 and ((152 <= prediction <= 269) or (283 <= prediction <= 294))): # hardcoded confidence threshold to 0.3
            dog_cat_found.append({"image": image_i, "score": score, "prediction": prediction, "subimage_coord": patch_coord[curr_index]})
        
        curr_index += 1
    
    # create bounding box around identified cat and dog
    bbox_creation = ImageDraw.Draw(input_image) 
    for found_object in dog_cat_found: 

        x_min, y_min, x_max, y_max = found_object["subimage_coord"]
        box_xyxy = [x_min, y_min, x_max, y_max] 
        bbox_creation.rectangle(xy=box_xyxy) # takes box coordinates in xyxy

        prediction_text = "dog"
        if (283 <= found_object["prediction"] <= 294):
            prediction_text = "cat"
        
        print(f"[INFO] Drawing box: {box_xyxy} (XYXY), label: {prediction_text}")
        label_location = (x_min + 5, y_min + 5)  #offset of label from the box 
        bbox_creation.text(label_location, f"{prediction_text}")

    input_image.save(f"{image_name}_baseline_image.png")


if __name__ == "__main__":

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'

    run_baseline("../2007_001239.jpg", "2007_image", device)