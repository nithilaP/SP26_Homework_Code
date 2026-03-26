import sys
import numpy as np
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

import json
import os
from ultralytics import YOLO

if __name__ == '__main__':

    ## COCO LOADER ##

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'
        
    # YOLO LOOP
    yolo_model = YOLO("yolov8n.pt")

    coco_val = '/content/val2017'
    annotation_file = '/content/annotations/instances_val2017.json'

    # set up mappings from coco json 
    dataset = json.load(open(annotation_file, 'r'))

    # creating index 
    print("creating index...")

    anns = {}
    imgs = {}
    cats = {}
    imgToAnns = {}

    # add annotations by annotation id & image_id -> list of annotations
    if 'annotations' in dataset: 
        for ann in dataset['annotations']:
            # CHECK IF IF CLAUSE BELOW NEEDED.
            if ann['image_id'] not in imgToAnns:
                imgToAnns[ann['image_id']] = []
            imgToAnns[ann['image_id']].append(ann)

            anns[ann['id']] = ann
    
    # add images by image_id
    if 'images' in dataset: 
        for img in dataset['images']:
            imgs[img['id']] = img 

    # add categories by category id
    if 'categories' in dataset: 
        for cat in dataset['categories']:
            cats[cat['id']] = cat

    # if "annotations" in dataset and "categories" in dataset: 
    #     for ann in dataset["annotations"]:
    #         catToImgs[ann["category_id"]].append(ann["image_id"])

    # For debugging
    all_image_ids = list(imgs.keys())

    print("number of images:", len(imgs))
    print("number of annotations:", len(anns))
    print("number of categories:", len(cats))
    
    print("index created!")

    # create mappings for categories 
    cat_id_to_name = {}
    cat_name_to_id = {}

    for cat_id in cats: 
        cat_id_to_name[cat_id] = cats[cat_id]["name"]
        cat_name_to_id[cats[cat_id]["name"]] = cat_id

    # go through all images: load image, get annotations, convert to bouding box 
    # for i in range(len(all_image_ids)):

    # FOR DEBUGGING: 
    for i in range(min(10, len(all_image_ids))):

        # get image info 
        curr_img = imgs[all_image_ids[i]]
        curr_image_path = os.path.join(coco_val, curr_img["file_name"])
        curr_image = Image.open(curr_image_path).convert("RGB")

        # DEBUG
        print("\nimage number:", i + 1)
        print("image id:", all_image_ids[i])
        print("file name:", curr_img["file_name"])
        print("image size:", curr_image.size)

        # ADD IN YOLO OUPUT PREDICTION
        yolo_output = yolo_model.predict(curr_image, verbose=False, device=device)
        yolo_result = yolo_output[0]

        yolo_bbox = []
        if len(yolo_result.boxes) > 0:
            yolo_bbox_i = yolo_result.boxes.xyxy.cpu().numpy()
            yolo_scores_i = yolo_result.conf.xyxy.cpu().numpy()
            yolo_labels_i = yolo_result.cls.xyxy.cpu().numpy()

            for j in range(len(yolo_bbox_i)):
                class_name = yolo_model.names[int(yolo_labels_i[j])]

                # get the class name from yolo -> coco category id
                if class_name in cat_name_to_id:

                    yolo_bbox.append({'bbox': [float(yolo_bbox_i[j][0]), float(yolo_bbox_i[j][1]), float(yolo_bbox_i[j][2]), float(yolo_bbox_i[j][3])], 
                                      'category_name': class_name, 
                                      'category_id': cat_name_to_id[class_name], 
                                      'score': float(yolo_scores_i[j])})


        curr_annotations = imgToAnns.get(all_image_ids[i], [])
        print("number of annotations:", len(curr_annotations))

        # convert boxes from coco format to [x_min, y_min, x_max, y_max]
        coco_bbox = [] 
        for curr_annotation in curr_annotations:

            if curr_annotation.get("iscrowd", 0) == 1: 
                continue 

            x, y, w, h = curr_annotation['bbox']

            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h

            category_id = curr_annotation['category_id']
            # category_name = cat_id_to_name[curr_annotation['category_id']]

            ## Bounding Box Item Format: 
            # bbox: [x_min, y_min, x_max, y_max]
            # category_name: (for yolo class name and coco dataset category comparison)
            # category_id: id from coco annotation
            coco_bbox.append({'bbox': [x_min, y_min, x_max, y_max], 'category_name': cat_id_to_name[category_id],'category_id': category_id})

            # DEBUGGING: 
            print("first few gt boxes:", coco_bbox[:3])
            print("first few yolo boxes:", yolo_bbox[:3])



