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
    baseline_model = resnet18(weights=ResNet18_Weights.DEFAULT) # DEFAULT = IMAGENET1K_V1
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

            # put image through model -> outputs 1000 scores
            model_output = baseline_model(input_batch)

            # model_output = unnormalized scores -> run softmas to get probabilities
            probabilities = F.softmax(model_output[0], dim=0)

            # get index with highest score / energy -> choose as prediction
            score, prediction = torch.max(probabilities, dim=0)
            score = score.item() # 
            prediction = prediction.item()

            print(f"{image_name}:: subimage_prediction: {prediction}| prediction_name: {ResNet18_Weights.DEFAULT.meta["categories"][prediction]}| score: {score}")


        # validate detected cat or dog
        # ImageNet Classes: https://github.com/pytorch/hub/blob/master/imagenet_classes.txt (for torch pretrained models -> ResNet18)
        # Dog: 152 (Chihuahua) to 269 (Mexican hairless)
        # Cat: 282 (tabby) to 286 (Egyptian cat)
        if (score > 0.3 and ((151 <= prediction <= 268) or (281 <= prediction <= 285))): # hardcoded confidence threshold to 0.3
            dog_cat_found.append({"image": image_i, "score": score, "prediction": prediction, "subimage_coord": patch_coord[curr_index], "image_label": ResNet18_Weights.DEFAULT.meta["categories"][prediction]})
        
        curr_index += 1
    
    # create bounding box around identified cat and dog
    bbox_creation = ImageDraw.Draw(input_image) 
    for found_object in dog_cat_found: 

        x_min, y_min, x_max, y_max = found_object["subimage_coord"]
        box_xyxy = [x_min, y_min, x_max, y_max] 
        bbox_creation.rectangle(xy=box_xyxy) # takes box coordinates in xyxy

        prediction_text = "dog"
        if (281 <= found_object["prediction"] <= 285):
            prediction_text = "cat"
        
        print(f"subimage: {box_xyxy}, label: {prediction_text}, prediction_name: {found_object["image_label"]}")
        label_location = (x_min + 5, y_min + 5)  #offset of label from the box 
        bbox_creation.text(label_location, f"{prediction_text} | {found_object["image_label"]}")

    input_image.save(f"{image_name}_baseline_image.png")

# Overlapping Patches Algorithm: Generate more patches with overlapping boundaries to capture center of detected object better
def overlapping_patches_implementation(input_image, image_name, device):

    # INPUT PROCESSING 

    # fetch & convert input 
    input_image = Image.open(input_image)
    # input_image = model_input.convert("RGB")

    # split image into 5 by 5 non-overlapping patches
    image_width = input_image.size[0]
    image_height = input_image.size[1]

    patch_width = image_width // 5 
    patch_height = image_height // 5
    patch_coord = [] # with overlapping patches, need to generate more patches
    subimages = [] # array w coordinates of each sub-images

    # generate sliding windows of patches and add to the list 
    starting_x = 0; 
    starting_y = 0; 

    while (starting_x <= image_width - patch_width):
        while (starting_y <= image_height - patch_height):

            patch_vals = (starting_x, starting_y, starting_x + patch_width, starting_y + patch_height);
            patch_coord.append(patch_vals)
            subimages.append(input_image.crop(patch_vals))

            starting_y += patch_height // 3 # move over 1/5 of a patch for the next patch
        starting_y = 0 # reset y once all vertical patches on that x axis are done
        starting_x += patch_width // 3 # move over 1/5 of a patch height for the next patch

    print(f"number of patches: {len(patch_coord)}")

    # for i in range(5): # i: top -> bottom 
    #     for j in range(5): # j: left -> right
    #         patch_vals = (patch_width * j, patch_height * i, patch_width * j + patch_width, patch_height * i + patch_height)
    #         patch_coord.append(patch_vals)
    #         subimages.append(input_image.crop(patch_vals))

    # set up model: https://docs.pytorch.org/vision/stable/models.html 
    baseline_model = resnet18(weights=ResNet18_Weights.DEFAULT) # DEFAULT = IMAGENET1K_V1
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

            # put image through model -> outputs 1000 scores
            model_output = baseline_model(input_batch)

            # model_output = unnormalized scores -> run softmas to get probabilities
            probabilities = F.softmax(model_output[0], dim=0)

            # get index with highest score / energy -> choose as prediction
            score, prediction = torch.max(probabilities, dim=0)
            score = score.item() # 
            prediction = prediction.item()

            print(f"{image_name}:: subimage_prediction: {prediction}| prediction_name: {ResNet18_Weights.DEFAULT.meta["categories"][prediction]}| score: {score}")


        # validate detected cat or dog
        # ImageNet Classes: https://github.com/pytorch/hub/blob/master/imagenet_classes.txt (for torch pretrained models -> ResNet18)
        # Dog: 152 (Chihuahua) to 269 (Mexican hairless)
        # Cat: 281 (tabby) to 293 (Cheetah)
        if (score > 0.5 and ((151 <= prediction <= 268) or (281 <= prediction <= 293))): # hardcoded confidence threshold to 0.3
            dog_cat_found.append({"image": image_i, "score": score, "prediction": prediction, "subimage_coord": patch_coord[curr_index], "image_label": ResNet18_Weights.DEFAULT.meta["categories"][prediction]})
        
        curr_index += 1
    
    # create bounding box around identified cat and dog
    bbox_creation = ImageDraw.Draw(input_image) 
    for found_object in dog_cat_found: 

        x_min, y_min, x_max, y_max = found_object["subimage_coord"]
        box_xyxy = [x_min, y_min, x_max, y_max] 
        bbox_creation.rectangle(xy=box_xyxy) # takes box coordinates in xyxy

        prediction_text = "dog"
        if (281 <= found_object["prediction"] <= 293):
            prediction_text = "cat"
        
        print(f"subimage: {box_xyxy}, label: {prediction_text}, prediction_name: {found_object['image_label']}")
        label_location = (x_min + 5, y_min + 5)  #offset of label from the box 
        bbox_creation.text(label_location, f"{prediction_text} | {found_object['image_label']}")

    input_image.save(f"{image_name}_overlapping_patches_image.png")

# Overlap and Merge: Overlappign Patches and Merge patches in same region w similar detection
def overlap_and_merge_implementation(input_image, image_name, device):

    # INPUT PROCESSING 

    # fetch & convert input 
    input_image = Image.open(input_image)
    # input_image = model_input.convert("RGB")

    # split image into 5 by 5 non-overlapping patches
    image_width = input_image.size[0]
    image_height = input_image.size[1]

    patch_width = image_width // 5 
    patch_height = image_height // 5
    patch_coord = [] # with overlapping patches, need to generate more patches
    subimages = [] # array w coordinates of each sub-images

    # generate sliding windows of patches and add to the list 
    starting_x = 0; 
    starting_y = 0; 

    while (starting_x <= image_width - patch_width):
        while (starting_y <= image_height - patch_height):

            patch_vals = (starting_x, starting_y, starting_x + patch_width, starting_y + patch_height);
            patch_coord.append(patch_vals)
            subimages.append(input_image.crop(patch_vals))

            starting_y += patch_height // 3 # move over 1/5 of a patch for the next patch
        starting_y = 0 # reset y once all vertical patches on that x axis are done
        starting_x += patch_width // 3 # move over 1/5 of a patch height for the next patch

    print(f"number of patches: {len(patch_coord)}")

    # for i in range(5): # i: top -> bottom 
    #     for j in range(5): # j: left -> right
    #         patch_vals = (patch_width * j, patch_height * i, patch_width * j + patch_width, patch_height * i + patch_height)
    #         patch_coord.append(patch_vals)
    #         subimages.append(input_image.crop(patch_vals))

    # set up model: https://docs.pytorch.org/vision/stable/models.html 
    baseline_model = resnet18(weights=ResNet18_Weights.DEFAULT) # DEFAULT = IMAGENET1K_V1
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

            # put image through model -> outputs 1000 scores
            model_output = baseline_model(input_batch)

            # model_output = unnormalized scores -> run softmas to get probabilities
            probabilities = F.softmax(model_output[0], dim=0)

            # get index with highest score / energy -> choose as prediction
            score, prediction = torch.max(probabilities, dim=0)
            score = score.item() # 
            prediction = prediction.item()

            print(f"{image_name}:: subimage_prediction: {prediction}| prediction_name: {ResNet18_Weights.DEFAULT.meta["categories"][prediction]}| score: {score}")


        # validate detected cat or dog
        # ImageNet Classes: https://github.com/pytorch/hub/blob/master/imagenet_classes.txt (for torch pretrained models -> ResNet18)
        # Dog: 152 (Chihuahua) to 269 (Mexican hairless)
        # Cat: 281 (tabby) to 293 (Cheetah)
        if (score > 0.3 and ((151 <= prediction <= 268) or (281 <= prediction <= 293))): # hardcoded confidence threshold to 0.3

            pred_label = "dog"
            if (281 <= prediction<= 293):
                pred_label = "cat"

            dog_cat_found.append({"image": image_i, "score": score, "prediction": prediction, "subimage_coord": patch_coord[curr_index], "image_label": ResNet18_Weights.DEFAULT.meta["categories"][prediction], "prediction_label": pred_label})
        
        curr_index += 1

    # merge boxes that are around each other
    merged_found=[]
    merged_found_subimages=[]
    already_merged = [False for i in dog_cat_found]
    for i, found_object in enumerate(dog_cat_found):

        if (already_merged[i] == True):
            continue;
        
        # set the merged coordinates as the current start box's
        merge_x_min, merge_y_min, merge_x_max, merge_y_max = found_object["subimage_coord"]
        
        # find center for current box for distance calc
        center_x = 0.5 * (merge_x_min + merge_x_max) 
        center_y = 0.5 * (merge_y_min + merge_y_max)

        # tracking var for loop
        x_min, y_min, x_max, y_max = found_object["subimage_coord"]
        curr_box_pred = found_object["prediction"]
        curr_box_pred_label = found_object["image_label"]
        curr_box_score = found_object["score"]
        for j in range(i+1, len(dog_cat_found)):

            # if already merged move on. 
            if (already_merged[j] == True):
                continue;
    
            if (found_object["prediction_label"] != dog_cat_found[j]["prediction_label"]):
                continue;
    
            # find center of j box for distance eq
            merge_x_j_min, merge_y_j_min, merge_x_j_max, merge_y_j_max = dog_cat_found[j]["subimage_coord"]
            center_j_x = 0.5 * (merge_x_j_min + merge_x_j_max) 
            center_j_y = 0.5 * (merge_y_j_min + merge_y_j_max)

            # distance eq 
            distance = ((center_x - center_j_x) ** 2 + (center_y - center_j_y) ** 2) ** 0.5

            if (distance < patch_width * 1.0):

                # do the merge
                # Update merge_x_min etc and center_x and center_y
                merge_x_min = min(merge_x_min, merge_x_j_min)
                merge_y_min = min(merge_y_min, merge_y_j_min)

                merge_x_max = max(merge_x_max, merge_x_j_max)
                merge_y_max = max(merge_y_max, merge_y_j_max)

                center_x = 0.5 * (merge_x_min + merge_x_max) 
                center_y = 0.5 * (merge_y_min + merge_y_max)

                # update merge tracker 
                already_merged[j] = True

                #
                if dog_cat_found[j]["score"] > curr_box_score:
                    curr_box_pred = dog_cat_found[j]["prediction"]
                    curr_box_pred_label = dog_cat_found[j]["image_label"]
                    curr_box_score = dog_cat_found[j]["score"]

        merged_vals = (merge_x_min, merge_y_min, merge_x_max, merge_y_max)

        merged_found.append({"score": curr_box_score, "prediction": curr_box_pred,
                             "subimage_coord": merged_vals, "image_label": curr_box_pred_label, 
                             "prediction_label": found_object["prediction_label"]})
        
        merged_found_subimages.append(input_image.crop(merged_vals))

        already_merged[i] = True # update for next few iterations

    print(f"Number of merged boxes: {len(merged_found)}")
    # redo the detection on this
    dog_cat_found_w_merged_box = []
    curr_index = 0
    for image_i in merged_found_subimages:
        # classify with the model
        with torch.no_grad(): 
            
            # preprocess the image
            input_tensor = preprocess_transform(image_i)
            input_batch = input_tensor.unsqueeze(0)
            input_batch = input_batch.to(device)

            # put image through model -> outputs 1000 scores
            model_output = baseline_model(input_batch)

            # model_output = unnormalized scores -> run softmas to get probabilities
            probabilities = F.softmax(model_output[0], dim=0)

            # get index with highest score / energy -> choose as prediction
            score, prediction = torch.max(probabilities, dim=0)
            score = score.item() # 
            prediction = prediction.item()

            print(f"{image_name}:: subimage_prediction: {prediction}| prediction_name: {ResNet18_Weights.DEFAULT.meta['categories'][prediction]}| score: {score}")


        # validate detected cat or dog
        # ImageNet Classes: https://github.com/pytorch/hub/blob/master/imagenet_classes.txt (for torch pretrained models -> ResNet18)
        # Dog: 152 (Chihuahua) to 269 (Mexican hairless)
        # Cat: 281 (tabby) to 293 (Cheetah)
        if (score > 0.8 and ((151 <= prediction <= 268) or (281 <= prediction <= 293))): # hardcoded confidence threshold to 0.3

            pred_label = "dog"
            if (281 <= prediction <= 293):
                pred_label = "cat"

            dog_cat_found_w_merged_box.append({"image": image_i, "score": score, "prediction": prediction, "subimage_coord": merged_found[curr_index]["subimage_coord"],
                                               "image_label": ResNet18_Weights.DEFAULT.meta['categories'][prediction], "prediction_label": pred_label})
        
        curr_index += 1

    # create bounding box around identified cat and dog
    bbox_creation = ImageDraw.Draw(input_image) 
    for found_object in dog_cat_found_w_merged_box: 

        x_min, y_min, x_max, y_max = found_object["subimage_coord"]
        box_xyxy = [x_min, y_min, x_max, y_max] 
        bbox_creation.rectangle(xy=box_xyxy) # takes box coordinates in xyxy

        prediction_text = "dog"
        if (281 <= found_object["prediction"] <= 293):
            prediction_text = "cat"
        
        print(f"subimage: {box_xyxy}, label: {prediction_text}, prediction_name: {found_object['image_label']}")
        label_location = (x_min + 5, y_min + 5)  #offset of label from the box 
        bbox_creation.text(label_location, f"{prediction_text} | {found_object['image_label']}")

    input_image.save(f"{image_name}_image.png")


if __name__ == "__main__":

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'

    # Determine categories: 
    # category = ResNet18_Weights.DEFAULT.meta["categories"]
    # for i in range(len(category)):
    #     print(i, category[i])

    # run_baseline("../2007_001239.jpg", "dog_image", device)
    # run_baseline("../2008_002152.jpg", "cat_image", device)

    # overlapping_patches_implementation("../2007_001239.jpg", "dog_image_overlap", device)
    # overlapping_patches_implementation("../2008_002152.jpg", "cat_image_overlap", device)

    overlap_and_merge_implementation("../2007_001239.jpg", "dog_image_overlap_and_merge", device)
    overlap_and_merge_implementation("../2008_002152.jpg", "cat_image_overlap_and_merge", device)