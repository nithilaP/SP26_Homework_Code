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

def performance_polling(val_link, annotations_link, device):



if __name__ == "__main__":

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'

    # load YOLO
    yolo_model = YOLO("yolov8n.pt")

    coco_val = "/content/val2017"
    coco_annotations = "/content/annotations/instances_val2017.json"

    # set up mappings from coco json 
    coco_file = open(coco_annotations, "r")
    coco_json = json.load(coco_file)
    coco_file.close()

    # coco dataset components 
    images = coco_json["images"]
    annotations = coco_json["annotations"]
    categories = coco_json["categories"]

    # need to map:
    #  each image id to its info 
    #  each image id to annotations for it
    image_mapping = {}
    all_image_ids = []
    for image_i in images: 
        image_mapping[image_i["id"]] = image_i
    
        # create list of image ids
        all_image_ids.append(image_i["id"])

    image_to_annotations = {}
    for annotation_i in annotations: 
        if annotation_i["image_id"] not in image_to_annotations:
            image_to_annotations[annotation_i["image_id"]] = []

        image_to_annotations[annotation_i["image_id"]].append(annotation_i)

    # create mappings for categories 
    category_id_to_name = {}
    category_name_to_id = {}

    for category in categories: 
        category_id_to_name[category["id"]] = category["name"]
        category_name_to_id[category["name"]] = category["id"]

    # go through all images: load image, get annotations, convert to bouding box 
    for i in range(len(all_image_ids)):

        curr_image_path = os.path.join(coco_val, image_mapping[all_image_ids[i]]["file_name"])
        curr_image = Image.open(curr_image_path).convert("RGB")

        yolo_output = yolo_model.predict(curr_image, verbose=False)
        yolo_result = yolo_output[0]

        yolo_bbox = []
        if (len(yolo_result.boxes) > 0):
            
            yolo_boxes = yolo_result.boxes.xyxy.cpu().numpy()
            yolo_scores = yolo_result.boxes.conf.cpu().numpy()
            yolo_labels = yolo_result.boxes.cls.cpu().numpy().astype(int)
            

        curr_annotations = image_to_annotations.get(all_image_ids[i], [])

        bounding_box = [] 
        for curr_annotation in curr_annotations:

            if curr_annotation.get("iscrowd", 0) == 1: 
                continue 

            x, y, w, h = curr_annotation["bbox"]

            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h

            category_id = curr_annotation["category_id"]
            category_name = category_id_to_name[category_id]

            bounding_box.append({"bbox": [x_min, y_min, x_max, y_max], "category_id": category_id,"category_name": category_name})



