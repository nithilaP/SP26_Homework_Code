import numpy as np
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

import json
import os
from ultralytics import YOLO

import random

def add_bbox(image, yolo_model):

    curr_image = Image.open(image)

    # inferencing
    yolo_output = yolo_model.predict(curr_image, verbose=False, device='cuda')
    yolo_result = yolo_output[0]

    # create bounding box around identified cat and dog
    bbox_creation = ImageDraw.Draw(curr_image) 
    if len(yolo_result.boxes) > 0:
        ## dim:
        #  yolo_bbox_i -> [b, 4]
        #  yolo_scores_i -> [b]
        #  yolo_labels_i -> [b]
        #  prediction: https://docs.ultralytics.com/tasks/detect/#dataset-format
        yolo_bbox_i = yolo_result.boxes.xyxy.cpu().numpy() # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        yolo_scores_i = yolo_result.boxes.conf.cpu().numpy() # confidence score of each box
        yolo_labels_i = yolo_result.boxes.cls.cpu().numpy() # class name of each box

        # convert yolo output into bbox list item
        for yolo_output_i in range(len(yolo_bbox_i)):

            # check if within threshold of 0.3
            if (yolo_scores_i[yolo_output_i] < 0.3):
                continue

            # class index value into class name for translation:
            #   https://stackoverflow.com/questions/77477793/class-ids-and-their-relevant-class-names-for-yolov8-model
            #   https://github.com/ultralytics/ultralytics/issues/1544
            class_index_val = yolo_labels_i[yolo_output_i]
            class_name = yolo_model.names[int(class_index_val)]

            x_min = yolo_bbox_i[yolo_output_i][0]
            y_min = yolo_bbox_i[yolo_output_i][1]
            x_max = yolo_bbox_i[yolo_output_i][2]
            y_max = yolo_bbox_i[yolo_output_i][3]

            box_xyxy = [x_min, y_min, x_max, y_max] 
            bbox_creation.rectangle(xy=box_xyxy) # takes box coordinates in xyxy

            label_location = (x_min + 5, y_min + 5)  #offset of label from the box 
            bbox_creation.text(label_location, f"{class_name} | {yolo_scores_i[yolo_output_i]:.2f}")

    curr_image.save(f"{image}_output.jpg")
    print(f"saved visualization to: {image}_output.jpg")

if __name__ == '__main__':

    ## DATA & MODEL LOADER ##

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # coco data path
    coco_dir = '/content/val2017'
    annotation_file = '/content/annotations/instances_val2017.json'
        
    # DELIVERABLE 5 MODEL: YOLOv8s-world
    yolo_model = YOLO("yolov8s-world.pt")

    # DATASET LOAD & SETUP #
    # COCO API __init__ & create_index inspo: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
    
    # set up mappings from coco json 
    dataset = json.load(open(annotation_file, 'r'))

    # dictionaries for populating
    anns = {}
    imgs = {}
    cats = {}
    imgToAnns = {}

    # add annotations by annotation id & image_id -> list of annotations
    if 'annotations' in dataset: 
        for ann in dataset['annotations']:

            if ann['image_id'] not in imgToAnns:
                imgToAnns[ann['image_id']] = []
            imgToAnns[ann['image_id']].append(ann)

            anns[ann['id']] = ann
    
    # add images by image_id
    if 'images' in dataset: 
        for img in dataset['images']:
            imgs[img['id']] = img
    all_image_ids = list(imgs.keys()) # for iteration. 

    # add categories by category id
    if 'categories' in dataset: 
        for cat in dataset['categories']:
            cats[cat['id']] = cat

    # create mappings for categories 
    cat_id_to_name = {}
    cat_name_to_id = {}

    for cat_id in cats: 
        cat_id_to_name[cat_id] = cats[cat_id]["name"]
        cat_name_to_id[cats[cat_id]["name"]] = cat_id

    # need to set classes
    #  https://docs.ultralytics.com/models/yolo-world/#set-prompts
    total_coco_classes = [] 
    for cat_id in sorted(list(cats.keys())):
        total_coco_classes.append(cat_id_to_name[cat_id])
    yolo_model.set_classes(total_coco_classes)

    # trackign variables 
    total_coco_data_bbox = []
    total_yolo_data_bbox = []

    # go through all images: load image, get annotations, convert to bounding box, save to list of bbox & image data
    for image_id_i in range(len(all_image_ids)):

        # get image info 
        curr_img = imgs[all_image_ids[image_id_i]]
        curr_image_path = os.path.join(coco_dir, curr_img["file_name"])
        curr_image = Image.open(curr_image_path)

        # populate list for annotations for this image id
        curr_annotations = []
        if (all_image_ids[image_id_i] in imgToAnns):
            curr_annotations = imgToAnns[all_image_ids[image_id_i]]
        
        # COCO BBOX CALC PER IMAGE
        # convert boxes from coco format -> [x_min, y_min, x_max, y_max]
        coco_bbox = [] 
        for curr_annotation in curr_annotations:

            # if curr_annotation.get("iscrowd", 0) == 1: 
            #     continue 

            x = curr_annotation['bbox'][0]
            y = curr_annotation['bbox'][1]
            width = curr_annotation['bbox'][2]
            height = curr_annotation['bbox'][3]

            # calculate bounding box value & values for coco_bbox list item 
            x_min = x
            x_max = x + width
            y_min = y
            y_max = y + height

            ## Bounding Box Item Format: 
            # bbox: [x_min, y_min, x_max, y_max]
            # category_name: (for yolo class name and coco dataset category comparison)
            # category_id: id from coco annotation
            # image_id: the associated image_id from iteration 
            coco_bbox.append({'bbox': [x_min, y_min, x_max, y_max], 'category_name': cat_id_to_name[curr_annotation['category_id']],
                              'category_id': curr_annotation['category_id'], 'image_id': all_image_ids[image_id_i],})

        # YOLO OUPUT PREDICTION & BOX CALC
        yolo_output = yolo_model.predict(curr_image, verbose=False, device=device)
        yolo_result = yolo_output[0]

        yolo_bbox = []
        if len(yolo_result.boxes) > 0:
            ## dim:
            #  yolo_bbox_i -> [b, 4]
            #  yolo_scores_i -> [b]
            #  yolo_labels_i -> [b]
            #  prediction: https://docs.ultralytics.com/tasks/detect/#dataset-format
            yolo_bbox_i = yolo_result.boxes.xyxy.cpu().numpy() # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            yolo_scores_i = yolo_result.boxes.conf.cpu().numpy() # confidence score of each box
            yolo_labels_i = yolo_result.boxes.cls.cpu().numpy() # class name of each box

            # convert yolo output into bbox list item
            for yolo_output_i in range(len(yolo_bbox_i)):

                # if yolo_scores_i[yolo_output_i] < 0.1:
                #     continue

                # class index value into class name for translation:
                #   https://stackoverflow.com/questions/77477793/class-ids-and-their-relevant-class-names-for-yolov8-model
                #   https://github.com/ultralytics/ultralytics/issues/1544
                class_index_val = yolo_labels_i[yolo_output_i]
                class_name = yolo_model.names[int(class_index_val)]

                # get the class name from yolo -> coco category id
                if class_name in cat_name_to_id:
                    yolo_bbox.append({'bbox': [yolo_bbox_i[yolo_output_i][0], yolo_bbox_i[yolo_output_i][1], yolo_bbox_i[yolo_output_i][2], yolo_bbox_i[yolo_output_i][3]], 
                                      'category_name': class_name, 'category_id': cat_name_to_id[class_name], 
                                      'score': yolo_scores_i[yolo_output_i], 'image_id': all_image_ids[image_id_i],})

        # for each image, add to total list
        for bbox in coco_bbox:
            total_coco_data_bbox.append(bbox)
        for bbox in yolo_bbox:
            total_yolo_data_bbox.append(bbox)

    print("total gt boxes:", len(total_coco_data_bbox))
    print("total yolo boxes:", len(total_yolo_data_bbox))

    ## mAP50 Calculation
    # https://www.v7labs.com/blog/mean-average-precision
    # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    '''
    1. Loop over all classes. 
    2. get the dataset coco bbox and yolo bbox items for each class 
    3. using confidence val, sort predictions to build curve
    4. find IoU for all coco bbox and yolo bbox that match in id, store box that has the highest IoU
    5. determine if tp or fp & set flag in array. 
    6. calculate precision and recall
    7. map out precision-recall curve & calculate area under curve 
    8. take map avg for all the classes 
    '''

    # YOLO mAP50 Calculation
    class_ap = [] # compute AP per class 
    for c in sorted(list(cats.keys())): 

        coco_classes = []
        for coco in total_coco_data_bbox:
            if coco['category_id'] == c:
                coco_classes.append(coco)

        yolo_classes = []
        for yolo in total_yolo_data_bbox:
            if yolo['category_id'] == c:
                yolo_classes.append(yolo)
        
        # check if coco_class empty for that category 
        if len(coco_classes) == 0:
            continue

        # 3. using confidence val, sort predictions to build curve
        # sorting code: Sort the detections based one score - https://github.com/ultralytics/yolov5/issues/6245
        # need reverse = True -> highest to lowest score
        yolo_curve = sorted(yolo_classes, key=lambda x: x['score'], reverse=True) 

        # tracking variables 
        accurate_pred_map = [] # for each value, append 1 if true pred, 0 if false
        coco_in_curve = [False] * len(coco_classes) # bool array to track which blocks have already been matched
        yolo_i = 0 # index tracker

        for yolo_prediction in yolo_curve:
            
            # tracking var for best IoU yolo prediction
            curr_highest_iou = 0.0
            curr_highest_coco_i = -1
            coco_j = 0 # index tracker

            for coco_val in coco_classes:

                if coco_val['image_id'] != yolo_prediction['image_id']: # not match
                    coco_j += 1
                    continue

                # get the values 
                coco_x_min = coco_val['bbox'][0]
                coco_y_min = coco_val['bbox'][1]
                coco_x_max = coco_val['bbox'][2]
                coco_y_max = coco_val['bbox'][3]

                yolo_x_min = yolo_prediction['bbox'][0]
                yolo_y_min = yolo_prediction['bbox'][1]
                yolo_x_max = yolo_prediction['bbox'][2]
                yolo_y_max = yolo_prediction['bbox'][3]

                # calculate union & intersection region
                overlap_x_min = max(coco_x_min, yolo_x_min)
                overlap_y_min = max(coco_y_min, yolo_y_min)

                overlap_x_max = min(coco_x_max, yolo_x_max)
                overlap_y_max = min(coco_y_max, yolo_y_max)

                overlap_area = max(0.0, overlap_x_max - overlap_x_min) * max(0.0, overlap_y_max - overlap_y_min)
                coco_area = max(0.0, coco_x_max - coco_x_min) * max(0.0, coco_y_max - coco_y_min) 
                yolo_area = max(0.0, yolo_x_max - yolo_x_min) * max(0.0, yolo_y_max - yolo_y_min)

                union_area = (coco_area + yolo_area) - overlap_area
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
            yolo_i += 1

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

        for i in range(len(precision)-1, 0, -1):
            precision[i-1] = max(precision[i-1], precision[i])

        # calculate the area under the curve. 
        ap = 0.0
        for i in range(len(recall) - 1):
            ap += precision[i + 1] * (recall[i + 1] - recall[i])

        class_ap.append(ap)
        
    mAP50 = np.mean(class_ap)
    print("YOLO mAP50:", mAP50)

    # YOLO measure latency: latency as the amount of time it takes from inputting an image to getting detection 
    #                      results with a batch size of 1, averaged across 100 random images from COCO
    # random.sample: 
    #   https://stackoverflow.com/questions/19084646/how-to-make-a-random-array-in-python
    #   https://www.geeksforgeeks.org/python/python-random-sample-function/
    images_for_latency = random.sample(all_image_ids, 100)

    yolo_latency_times = []
    for latency_image_id in images_for_latency:

        # get image info 
        curr_img = imgs[latency_image_id]
        curr_image_path = os.path.join(coco_dir, curr_img["file_name"])
        curr_image = Image.open(curr_image_path)
        # curr_image = curr_image.convert("RGB") # fix: now every image = 3 channels

        torch.cuda.synchronize() # sync gpu ops

        # measure time elapsed 
        start = time.time() # start timer
        yolo_model.predict(curr_image, verbose=False, device=device) # inferencing
        torch.cuda.synchronize() # sync gpu ops
        end = time.time() # end timer

        yolo_latency_times.append(end-start)

    avg_yolo_latency = np.mean(yolo_latency_times)
    print("YOLO latency:", avg_yolo_latency)
                
    # add bounding boxes around cat and dog image 
    dog_path = "../2007_001239.jpg"
    cat_path = "../2008_002152.jpg"

    ex_model = YOLO("yolov8s-world.pt")
    ex_classes = ["dog", "cat"]
    ex_model.set_classes(ex_classes)

    add_bbox(dog_path, ex_model)
    add_bbox(cat_path, ex_model)