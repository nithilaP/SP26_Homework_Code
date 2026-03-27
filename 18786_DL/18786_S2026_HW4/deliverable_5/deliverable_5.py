import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

import json
import os
from ultralytics import YOLO

import random

def calc_mAP50(cats, total_coco_data_bbox, total_model_data_bbox):
    class_ap = [] # compute AP per class 
    for c in sorted(list(cats.keys())): 

        coco_classes = []
        for coco in total_coco_data_bbox:
            if coco['category_id'] == c:
                coco_classes.append(coco)

        model_classes = []
        for yolo in total_model_data_bbox:
            if yolo['category_id'] == c:
                model_classes.append(yolo)
        
        # check if coco_class empty for that category 
        if len(coco_classes) == 0:
            continue

        # 3. using confidence val, sort predictions to build curve
        # sorting code: Sort the detections based one score - https://github.com/ultralytics/yolov5/issues/6245
        # need reverse = True -> highest to lowest score
        model_curve = sorted(model_classes, key=lambda x: x['score'], reverse=True) 

        # tracking variables 
        accurate_pred_map = [] # for each value, append 1 if true pred, 0 if false
        coco_in_curve = [False] * len(coco_classes) # bool array to track which blocks have already been matched
        model_i = 0 # index tracker

        for model_prediction in model_curve:
            
            # tracking var for best IoU yolo prediction
            curr_highest_iou = 0.0
            curr_highest_coco_i = -1
            coco_j = 0 # index tracker

            for coco_val in coco_classes:

                if coco_val['image_id'] != model_prediction['image_id']: # not match
                    coco_j += 1
                    continue

                # get the values 
                coco_x_min = coco_val['bbox'][0]
                coco_y_min = coco_val['bbox'][1]
                coco_x_max = coco_val['bbox'][2]
                coco_y_max = coco_val['bbox'][3]

                model_x_min = model_prediction['bbox'][0]
                model_y_min = model_prediction['bbox'][1]
                model_x_max = model_prediction['bbox'][2]
                model_y_max = model_prediction['bbox'][3]

                # calculate union & intersection region
                overlap_x_min = max(coco_x_min, model_x_min)
                overlap_y_min = max(coco_y_min, model_y_min)

                overlap_x_max = min(coco_x_max, model_x_max)
                overlap_y_max = min(coco_y_max, model_y_max)

                overlap_area = max(0.0, overlap_x_max - overlap_x_min) * max(0.0, overlap_y_max - overlap_y_min)
                coco_area = max(0.0, coco_x_max - coco_x_min) * max(0.0, coco_y_max - coco_y_min) 
                model_area = max(0.0, model_x_max - model_x_min) * max(0.0, model_y_max - model_y_min)

                union_area = (coco_area + model_area) - overlap_area
                if (union_area >0):
                    IoU = overlap_area / union_area
                else: 
                    IoU = 0.0

                # track best IoU 
                if (IoU > curr_highest_iou):
                    curr_highest_iou = IoU
                    curr_highest_coco_i = coco_j
                
                coco_j += 1

            # update true_prediction and false_prediction 
            if (curr_highest_coco_i !=-1) and (curr_highest_iou >= 0.5) and (coco_in_curve[curr_highest_coco_i] != True):
                accurate_pred_map.append(1)
                coco_in_curve[curr_highest_coco_i] = True # update bool as not available
            else:
                accurate_pred_map.append(0)

            # update trackign var 
            model_i += 1

        # determine true positive and false positive from accurate_pred_map
        true_positive = np.array(accurate_pred_map)
        tp = np.cumsum(true_positive)

        false_positive = 1 - np.array(accurate_pred_map)
        fp = np.cumsum(false_positive)

        precision = tp / (tp + fp + 1e-20)
        recall = tp 
        if (len(coco_classes) != 0):
            recall = tp / len(coco_classes)

        # add vals at the end for bounds. (end points)
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([1.0], precision, [0.0]))

        # 
        for i in range(len(precision)-1, 0, -1):
            precision[i-1] = max(precision[i-1], precision[i])

        # calculate the area under the curve. 
        ap = 0.0
        for i in range(len(recall) - 1):
            ap += precision[i + 1] * (recall[i + 1] - recall[i])

        class_ap.append(ap)
        
    mAP50 = np.mean(class_ap)
    print("YOLO mAP50:", mAP50)

    return mAP50

def bbox_box(image, model, output_classes):

    curr_image = Image.open(image)
    # curr_image = curr_image.convert("RGB") # fix: now every image = 3 channels

    # inferencing

if __name__ == '__main__':
