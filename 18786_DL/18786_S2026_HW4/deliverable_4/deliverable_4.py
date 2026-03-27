import sys
import numpy as np
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

import json
import os
from ultralytics import YOLO

import random

def get_image_path(imgs, image_id, coco_val):
    curr_img_info = imgs[image_id]
    file_name = curr_img_info["file_name"]

    if isinstance(file_name, dict):
        print("BAD curr_img_info:", curr_img_info)
        if "file_name" in file_name:
            file_name = file_name["file_name"]
        elif "name" in file_name:
            file_name = file_name["name"]
        else:
            raise TypeError(f"file_name is not a valid string: {file_name}")

    if not isinstance(file_name, str):
        raise TypeError(f"Expected file_name to be str, got {type(file_name)}: {file_name}")

    return os.path.join(coco_val, file_name)

if __name__ == '__main__':

    ## COCO LOADER ##

    # set up device 
    device = 'cpu'
    if torch.cuda.is_available():
        # device = torch.cuda.current_device()
        device = 'cuda'
        
    # YOLO LOAD
    yolo_model = YOLO("yolov8n.pt")

    # FASTER RCNN LOAD -> structure from deliverable 3
    faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    faster_rcnn_model.to(device)
    faster_rcnn_model.eval()
    faster_rcnn_categories = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

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

    # trackign variables 
    total_coco_data_bbox = []
    total_yolo_data_bbox = []
    total_faster_rcnn_data_bbox = []

    # go through all images: load image, get annotations, convert to bouding box 
    # for i in range(len(all_image_ids)):

    eval_image_ids = all_image_ids[:min(100, len(all_image_ids))]
    for i in range(min(100, len(eval_image_ids))):

        # get image info 
        # curr_img_info = imgs[all_image_ids[i]]
        # curr_image_path = os.path.join(coco_val, curr_img_info["file_name"])
        image_id = imgs[eval_image_ids[i]]
        curr_image_path = get_image_path(imgs, image_id, coco_val)

        curr_image = Image.open(curr_image_path).convert("RGB")

        # DEBUG
        # print("\nimage number:", i + 1)
        # print("image id:", all_image_ids[i])
        # print("file name:", curr_img_info["file_name"])
        # print("image size:", curr_image.size)
        # if (i + 1) % 100 == 0:
        #     print("processed:", i + 1)

        # ADD IN YOLO OUPUT PREDICTION
        yolo_output = yolo_model.predict(curr_image, verbose=False, device=device)
        yolo_result = yolo_output[0]

        yolo_bbox = []
        if len(yolo_result.boxes) > 0:
            ## dim:
            #  yolo_bbox_i -> [b, 4]
            #  yolo_scores_i -> [b]
            #  yolo_labels_i -> [b]
            yolo_bbox_i = yolo_result.boxes.xyxy.cpu().numpy() 
            yolo_scores_i = yolo_result.boxes.conf.cpu().numpy()
            yolo_labels_i = yolo_result.boxes.cls.cpu().numpy()

            for j in range(len(yolo_bbox_i)):
                class_name = yolo_model.names[int(yolo_labels_i[j])]

                # get the class name from yolo -> coco category id
                if class_name in cat_name_to_id:

                    yolo_bbox.append({'bbox': [float(yolo_bbox_i[j][0]), float(yolo_bbox_i[j][1]), float(yolo_bbox_i[j][2]), float(yolo_bbox_i[j][3])], 
                                      'category_name': class_name, 
                                      'category_id': cat_name_to_id[class_name], 
                                      'score': float(yolo_scores_i[j]), 'image_id': all_image_ids[i],})


        curr_annotations = imgToAnns.get(all_image_ids[i], [])
        # print("number of annotations:", len(curr_annotations))
        if (i + 1) % 100 == 0:
            print("processed:", i + 1)

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
            coco_bbox.append({'bbox': [x_min, y_min, x_max, y_max], 'category_name': cat_id_to_name[category_id],'category_id': category_id, 'image_id': all_image_ids[i],})


        # do the faster_rcnn prediction 
        curr_img_tensor = transforms.ToTensor()(curr_image)
        curr_img_tensor = curr_img_tensor.to(device)

        with torch.no_grad():
            faster_rcnn_out = faster_rcnn_model([curr_img_tensor])[0]

        faster_rcnn_bbox = [] 
        if len(faster_rcnn_out['boxes']) > 0:
            ## dim:
            #  faster_rcnn_bbox_i -> [b, 4]
            #  faster_rcnn_scores_i -> [b]
            #  faster_rcnn_labels_i -> [b]
            faster_rcnn_bbox_i = faster_rcnn_out['boxes'].detach().cpu().numpy()
            faster_rcnn_scores_i = faster_rcnn_out['scores'].detach().cpu().numpy()
            faster_rcnn_labels_i = faster_rcnn_out['labels'].detach().cpu().numpy()

            for j in range(len(faster_rcnn_bbox_i)):
                class_name = faster_rcnn_categories[int(faster_rcnn_labels_i[j])]

                # skip empty labels
                if class_name == 'N/A':
                    continue

                # get the class name from yolo -> coco category id
                if class_name in cat_name_to_id:

                    faster_rcnn_bbox.append({'bbox': [float(faster_rcnn_bbox_i[j][0]), float(faster_rcnn_bbox_i[j][1]), float(faster_rcnn_bbox_i[j][2]), float(faster_rcnn_bbox_i[j][3])], 
                                      'category_name': class_name, 
                                      'category_id': cat_name_to_id[class_name], 
                                      'score': float(faster_rcnn_scores_i[j]), 'image_id': all_image_ids[i],})

        # for each image, add to total list
        for bbox in coco_bbox:
            total_coco_data_bbox.append(bbox)
        for bbox in yolo_bbox:
            total_yolo_data_bbox.append(bbox)
        for bbox in faster_rcnn_bbox:
            total_faster_rcnn_data_bbox.append(bbox)

    print("total gt boxes:", len(total_coco_data_bbox))
    print("total yolo boxes:", len(total_yolo_data_bbox))
    print("total faster rcnn boxes:", len(total_faster_rcnn_data_bbox))
    # DEBUGGING: 
    # print("first few gt boxes:", coco_bbox[:3])
    # print("first few yolo boxes:", yolo_bbox[:3])

    ## do the mAP50 Calculation 
    '''
    1. Loop over all classes. 
    2. get the dataset coco bbox and yolo bbox items for each class 
    3. using confidence val, sort predictions to build curve
    4. find IoU for all coco bbox and yolo bbox that match in id, store box that has the highest IoU
    5. determine if tp or fp & set flag in array. 
    6. calculate precision 
    7. map out precision curve & calculate area under curve 
    8. take map avg for all the classes 
    '''
    # YOLO MAP50
    class_avg_prec = [] # compute AP per class 
    acc_AP = 0
    valid_classes = 0
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
        # yolo_scores = []
        # yolo_curve = sorted(yolo_classes, reverse=True)
        yolo_curve = sorted(yolo_classes, key=lambda x: x['score'], reverse=True)

        coco_in_curve = [False] * len(coco_classes) # bool array to track which blocks have already been matched
        true_positive = [0] * len(yolo_curve)
        false_positive = [0] * len(yolo_curve)
        # true_positive_num = 0
        # false_positive_num = 0

        yolo_i = 0 # index tracker
        for yolo_prediction in yolo_curve:

            curr_highest_iou = 0.0
            curr_highest_coco_i = -1

            coco_j = 0 # index tracker
            for coco_val in coco_classes:

                if coco_val['image_id'] != yolo_prediction['image_id']:
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

                # track best 
                if (IoU > curr_highest_iou):
                    curr_highest_iou = IoU
                    curr_highest_coco_i = coco_j
                
                coco_j += 1

            # update true_prediction and false_prediction 
            if (curr_highest_coco_i !=-1) and (curr_highest_iou >= 0.5) and (coco_in_curve[curr_highest_coco_i] != True):
                true_positive[yolo_i] = True
                false_positive[yolo_i] = False
                # true_positive_num += 1
                coco_in_curve[curr_highest_coco_i] = True
            else: 
                true_positive[yolo_i] = False
                false_positive[yolo_i] = True
                # false_positive_num += 1

            # update trackign car 
            yolo_i += 1


        cummalative_sum_truep = np.cumsum(true_positive)
        cummalative_sum_falsep = np.cumsum(false_positive)
        precision = cummalative_sum_truep / (cummalative_sum_truep + cummalative_sum_falsep + 1e-8)
        recall = cummalative_sum_truep / len(coco_classes)

        #  CHECK!!!
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([1.0], precision, [0.0]))

        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        ap = 0.0
        for i in range(len(recall) - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]

        class_avg_prec.append(ap)
        acc_AP += ap
        valid_classes += 1

    mAP50 = acc_AP / valid_classes
    print("YOLO mAP50:", mAP50)


    # FASTER RCNN MAP50
    class_avg_prec = [] # compute AP per class 
    acc_AP = 0
    valid_classes = 0
    for c in sorted(list(cats.keys())): 

        coco_classes = []
        for coco in total_coco_data_bbox:
            if coco['category_id'] == c:
                coco_classes.append(coco)

        faster_rcnn_classes = []
        for yolo in total_faster_rcnn_data_bbox:
            if yolo['category_id'] == c:
                faster_rcnn_classes.append(yolo)
        
        # check if coco_class empty for that category 
        if len(coco_classes) == 0:
            continue

        # 3. using confidence val, sort predictions to build curve
        # faster_rcnn_scores = []
        # faster_rcnn_curve = sorted(faster_rcnn_classes, reverse=True)
        faster_rcnn_curve = sorted(faster_rcnn_classes, key=lambda x: x['score'], reverse=True)

        coco_in_curve = [False] * len(coco_classes) # bool array to track which blocks have already been matched
        true_positive = [0] * len(faster_rcnn_curve)
        false_positive = [0] * len(faster_rcnn_curve)
        # true_positive_num = 0
        # false_positive_num = 0

        faster_rcnn_i = 0 # index tracker
        for faster_rcnn_prediction in faster_rcnn_curve:

            curr_highest_iou = 0.0
            curr_highest_coco_i = -1

            coco_j = 0 # index tracker
            for coco_val in coco_classes:

                if coco_val['image_id'] != faster_rcnn_prediction['image_id']:
                    coco_j += 1
                    continue

                # get the values 
                
                coco_x_min = coco_val['bbox'][0]
                coco_y_min = coco_val['bbox'][1]
                coco_x_max = coco_val['bbox'][2]
                coco_y_max = coco_val['bbox'][3]

                faster_rcnn_x_min = faster_rcnn_prediction['bbox'][0]
                faster_rcnn_y_min = faster_rcnn_prediction['bbox'][1]
                faster_rcnn_x_max = faster_rcnn_prediction['bbox'][2]
                faster_rcnn_y_max = faster_rcnn_prediction['bbox'][3]

                overlap_x_min = max(coco_x_min, faster_rcnn_x_min)
                overlap_y_min = max(coco_y_min, faster_rcnn_y_min)

                overlap_x_max = min(coco_x_max, faster_rcnn_x_max)
                overlap_y_max = min(coco_y_max, faster_rcnn_y_max)

                overlap_area = max(0.0, overlap_x_max - overlap_x_min) * max(0.0, overlap_y_max - overlap_y_min) 
                coco_area = max(0.0, coco_x_max - coco_x_min) * max(0.0, coco_y_max - coco_y_min) 
                faster_rcnn_area = max(0.0, faster_rcnn_x_max - faster_rcnn_x_min) * max(0.0, faster_rcnn_y_max - faster_rcnn_y_min) 
                union_area = (coco_area + faster_rcnn_area) - overlap_area
                if (union_area >0):
                    IoU = overlap_area / union_area
                else: 
                    IoU = 0.0

                # track best 
                if (IoU > curr_highest_iou):
                    curr_highest_iou = IoU
                    curr_highest_coco_i = coco_j
                
                coco_j += 1

            # update true_prediction and false_prediction 
            if (curr_highest_coco_i !=-1) and (curr_highest_iou >= 0.5) and (coco_in_curve[curr_highest_coco_i] != True):
                true_positive[faster_rcnn_i] = True
                false_positive[faster_rcnn_i] = False
                # true_positive_num += 1
                coco_in_curve[curr_highest_coco_i] = True
            else: 
                true_positive[faster_rcnn_i] = False
                false_positive[faster_rcnn_i] = True
                # false_positive_num += 1

            # update trackign car 
            faster_rcnn_i += 1


        cummalative_sum_truep = np.cumsum(true_positive)
        cummalative_sum_falsep = np.cumsum(false_positive)
        precision = cummalative_sum_truep / (cummalative_sum_truep + cummalative_sum_falsep + 1e-8)
        recall = cummalative_sum_truep / len(coco_classes)

        #  CHECK!!!
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([1.0], precision, [0.0]))

        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        ap = 0.0
        for i in range(len(recall) - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]

        class_avg_prec.append(ap)
        acc_AP += ap
        valid_classes += 1

    mAP50 = acc_AP / valid_classes
    print("Faster RCNN mAP50:", mAP50)

    # YOLO measure latency: latency as the amount of time it takes from inputting an image to getting detection 
    #                      results with a batch size of 1, averaged across 100 random images from COCO
    random.seed(0)
    # images_for_latency = random.sample(all_image_ids, 100)

    latency_count = min(100, len(eval_image_ids))
    images_for_latency = random.sample(eval_image_ids, latency_count)

    yolo_latency_times = []
    for latency_image_id in images_for_latency:

        # get image info 
        curr_img_info = imgs[latency_image_id]
        # curr_image_path = os.path.join(coco_val, curr_img_info["file_name"])
        # curr_image = Image.open(curr_image_path).convert("RGB")

        curr_image_path = get_image_path(imgs, latency_image_id, coco_val)

        # file_name = curr_img_info["file_name"]

        # # debug / safety
        # if isinstance(file_name, dict):
        #     print("BAD curr_img_info:", curr_img_info)
        #     # try common nested forms
        #     if "file_name" in file_name:
        #         file_name = file_name["file_name"]
        #     elif "name" in file_name:
        #         file_name = file_name["name"]
        #     else:
        #         raise TypeError(f"file_name is still not usable: {file_name}")

        # if not isinstance(file_name, str):
        #     raise TypeError(f"file_name must be a string, got {type(file_name)} -> {file_name}")

        # curr_image_path = os.path.join(coco_val, file_name)
        curr_image = Image.open(curr_image_path).convert("RGB")

        # ADDED FOR GPU USAGE
        if device  == 'cuda': 
            torch.cuda.synchronize()

        # measure time elapsed 
        start = time.time()

        yolo_model.predict(curr_image, verbose=False, device=device)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.time()

        curr_latency = end - start
        yolo_latency_times.append(curr_latency)

    avg_yolo_latency_times = np.mean(yolo_latency_times)

    print("YOLO average latency (seconds):", avg_yolo_latency_times)
    print("YOLO average latency (milliseconds):", avg_yolo_latency_times * 1000.0)
                
    # FASTER RCCN measure latency: latency as the amount of time it takes from inputting an image to getting detection 
    #                      results with a batch size of 1, averaged across 100 random images from COCO
    random.seed(0)
    images_for_latency = random.sample(all_image_ids, 100)

    faster_rcnn_latency_times = []
    for latency_image_id in images_for_latency:

        # get image info 
        curr_img_info = imgs[latency_image_id]
        curr_image_path = os.path.join(coco_val, curr_img_info["file_name"])
        curr_image = Image.open(curr_image_path).convert("RGB")

        # ADDED FOR GPU USAGE
        if device  == 'cuda': 
            torch.cuda.synchronize()

        # measure time elapsed 
        start = time.time()

        # get predictions from model 
        curr_img_tensor = transforms.ToTensor()(curr_image)
        curr_img_tensor = curr_img_tensor.to(device)

        with torch.no_grad():
            faster_rcnn_out = faster_rcnn_model([curr_img_tensor])[0]

        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.time()

        curr_latency = end - start
        faster_rcnn_latency_times.append(curr_latency)

    avg_faster_rcnn_latency_times = np.mean(faster_rcnn_latency_times)

    print("Faster RCNN average latency (seconds):", avg_faster_rcnn_latency_times)
    print("Faster RCNN  average latency (milliseconds):", avg_faster_rcnn_latency_times * 1000.0)
                




        




