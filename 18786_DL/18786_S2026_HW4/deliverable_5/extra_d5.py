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


def compute_map50(cats, total_coco_data_bbox, total_pred_data_bbox):
    class_avg_prec = []
    acc_AP = 0.0
    valid_classes = 0

    for c in sorted(list(cats.keys())):

        coco_classes = []
        for coco in total_coco_data_bbox:
            if coco['category_id'] == c:
                coco_classes.append(coco)

        pred_classes = []
        for pred in total_pred_data_bbox:
            if pred['category_id'] == c:
                pred_classes.append(pred)

        if len(coco_classes) == 0:
            continue

        pred_curve = sorted(pred_classes, key=lambda x: x['score'], reverse=True)

        coco_in_curve = [False] * len(coco_classes)
        true_positive = [0] * len(pred_curve)
        false_positive = [0] * len(pred_curve)

        pred_i = 0
        for pred_box in pred_curve:
            curr_highest_iou = 0.0
            curr_highest_coco_i = -1

            coco_j = 0
            for coco_val in coco_classes:
                if coco_val['image_id'] != pred_box['image_id']:
                    coco_j += 1
                    continue

                coco_x_min = coco_val['bbox'][0]
                coco_y_min = coco_val['bbox'][1]
                coco_x_max = coco_val['bbox'][2]
                coco_y_max = coco_val['bbox'][3]

                pred_x_min = pred_box['bbox'][0]
                pred_y_min = pred_box['bbox'][1]
                pred_x_max = pred_box['bbox'][2]
                pred_y_max = pred_box['bbox'][3]

                overlap_x_min = max(coco_x_min, pred_x_min)
                overlap_y_min = max(coco_y_min, pred_y_min)
                overlap_x_max = min(coco_x_max, pred_x_max)
                overlap_y_max = min(coco_y_max, pred_y_max)

                overlap_area = max(0.0, overlap_x_max - overlap_x_min) * max(0.0, overlap_y_max - overlap_y_min)
                coco_area = max(0.0, coco_x_max - coco_x_min) * max(0.0, coco_y_max - coco_y_min)
                pred_area = max(0.0, pred_x_max - pred_x_min) * max(0.0, pred_y_max - pred_y_min)

                union_area = (coco_area + pred_area) - overlap_area
                if union_area > 0:
                    iou = overlap_area / union_area
                else:
                    iou = 0.0

                if iou > curr_highest_iou:
                    curr_highest_iou = iou
                    curr_highest_coco_i = coco_j

                coco_j += 1

            if (curr_highest_coco_i != -1) and (curr_highest_iou >= 0.5) and (coco_in_curve[curr_highest_coco_i] != True):
                true_positive[pred_i] = True
                false_positive[pred_i] = False
                coco_in_curve[curr_highest_coco_i] = True
            else:
                true_positive[pred_i] = False
                false_positive[pred_i] = True

            pred_i += 1

        cummalative_sum_truep = np.cumsum(true_positive)
        cummalative_sum_falsep = np.cumsum(false_positive)

        precision = cummalative_sum_truep / (cummalative_sum_truep + cummalative_sum_falsep + 1e-8)
        recall = cummalative_sum_truep / len(coco_classes)

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

    if valid_classes == 0:
        return 0.0

    return acc_AP / valid_classes

def draw_world_boxes(image_path, model, prompt_classes, output_path, device='cpu', conf_threshold=0.25):
    image = Image.open(image_path).convert("RGB")

    # set text prompts
    model.set_classes(prompt_classes)

    results = model.predict(image, verbose=False, device=device)
    result = results[0]

    draw = ImageDraw.Draw(image)

    if len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            if scores[i] < conf_threshold:
                continue

            x1, y1, x2, y2 = boxes[i]
            class_name = model.names[int(labels[i])]
            label_text = f"{class_name}: {scores[i]:.2f}"

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 12)), label_text, fill="red")

    image.save(output_path)
    print("saved visualization to:", output_path)

if __name__ == '__main__':

    # set up device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # DELIVERABLE 5 MODEL: YOLOv8s-world
    yolo_world_model = YOLO("yolov8s-world.pt")

    coco_dir = '/content/val2017'
    annotation_file = '/content/annotations/instances_val2017.json'

    # load COCO annotations
    dataset = json.load(open(annotation_file, 'r'))

    print("creating index...")

    anns = {}
    imgs = {}
    cats = {}
    imgToAnns = {}

    if 'annotations' in dataset:
        for ann in dataset['annotations']:
            if ann['image_id'] not in imgToAnns:
                imgToAnns[ann['image_id']] = []
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in dataset:
        for img in dataset['images']:
            imgs[img['id']] = img

    if 'categories' in dataset:
        for cat in dataset['categories']:
            cats[cat['id']] = cat

    all_image_ids = list(imgs.keys())

    print("number of images:", len(imgs))
    print("number of annotations:", len(anns))
    print("number of categories:", len(cats))
    print("index created!")

    cat_id_to_name = {}
    cat_name_to_id = {}

    for cat_id in cats:
        cat_id_to_name[cat_id] = cats[cat_id]["name"]
        cat_name_to_id[cats[cat_id]["name"]] = cat_id

    # COCO prompts for YOLO-world
    coco_class_names = []
    for cat_id in sorted(list(cats.keys())):
        coco_class_names.append(cat_id_to_name[cat_id])

    # IMPORTANT: set open-vocabulary prompts to COCO class names
    yolo_world_model.set_classes(coco_class_names)

    total_coco_data_bbox = []
    total_yolo_world_data_bbox = []

    # evaluate over all COCO val images
    for i in range(len(all_image_ids)):
        curr_img = imgs[all_image_ids[i]]
        curr_image_path = os.path.join(coco_dir, curr_img["file_name"])
        curr_image = Image.open(curr_image_path).convert("RGB")

        # YOLO-WORLD prediction
        yolo_output = yolo_world_model.predict(curr_image, verbose=False, device=device)
        yolo_result = yolo_output[0]

        yolo_world_bbox = []
        if len(yolo_result.boxes) > 0:
            yolo_bbox_i = yolo_result.boxes.xyxy.cpu().numpy()
            yolo_scores_i = yolo_result.boxes.conf.cpu().numpy()
            yolo_labels_i = yolo_result.boxes.cls.cpu().numpy()

            for j in range(len(yolo_bbox_i)):
                class_name = yolo_world_model.names[int(yolo_labels_i[j])]

                if class_name in cat_name_to_id:
                    yolo_world_bbox.append({
                        'bbox': [
                            float(yolo_bbox_i[j][0]),
                            float(yolo_bbox_i[j][1]),
                            float(yolo_bbox_i[j][2]),
                            float(yolo_bbox_i[j][3])
                        ],
                        'category_name': class_name,
                        'category_id': cat_name_to_id[class_name],
                        'score': float(yolo_scores_i[j]),
                        'image_id': all_image_ids[i],
                    })

        curr_annotations = imgToAnns.get(all_image_ids[i], [])

        if (i + 1) % 100 == 0:
            print("processed:", i + 1)

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

            coco_bbox.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'category_name': cat_id_to_name[category_id],
                'category_id': category_id,
                'image_id': all_image_ids[i],
            })

        for bbox in coco_bbox:
            total_coco_data_bbox.append(bbox)
        for bbox in yolo_world_bbox:
            total_yolo_world_data_bbox.append(bbox)

    print("total gt boxes:", len(total_coco_data_bbox))
    print("total yolo world boxes:", len(total_yolo_world_data_bbox))

    yolo_world_map50 = compute_map50(cats, total_coco_data_bbox, total_yolo_world_data_bbox)
    print("YOLOv8s-world mAP50:", yolo_world_map50)

    # OPTIONAL: latency for YOLOv8s-world too
    random.seed(0)
    images_for_latency = random.sample(all_image_ids, 100)

    yolo_world_latency_times = []
    for latency_image_id in images_for_latency:
        curr_img = imgs[latency_image_id]
        curr_image_path = os.path.join(coco_dir, curr_img["file_name"])
        curr_image = Image.open(curr_image_path).convert("RGB")

        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        yolo_world_model.predict(curr_image, verbose=False, device=device)

        if device == 'cuda':
            torch.cuda.synchronize()

        end = time.time()
        yolo_world_latency_times.append(end - start)

    avg_yolo_world_latency = np.mean(yolo_world_latency_times)
    print("YOLOv8s-world average latency (seconds):", avg_yolo_world_latency)
    print("YOLOv8s-world average latency (milliseconds):", avg_yolo_world_latency * 1000.0)

    # ---------------------------------------------------
    # PART 2: cat/dog prompt on two images
    # Replace these with your actual two image paths
    # ---------------------------------------------------

    # separate fresh model for cat/dog visualization
    catdog_model = YOLO("yolov8s-world.pt")
    catdog_device = "cpu"

    catdog_prompt_classes = ["cat", "dog"]

    dog_path = "../2007_001239.jpg"
    cat_path = "../2008_002152.jpg"

    if os.path.exists(dog_path):
        draw_world_boxes(
            image_path=dog_path,
            model=catdog_model,
            prompt_classes=catdog_prompt_classes,
            output_path="/content/dog_world_output.jpg",
            device=catdog_device,
            conf_threshold=0.25
        )
    else:
        print("could not find:", dog_path)

    if os.path.exists(cat_path):
        draw_world_boxes(
            image_path=cat_path,
            model=catdog_model,
            prompt_classes=catdog_prompt_classes,
            output_path="/content/cat_world_output.jpg",
            device=catdog_device,
            conf_threshold=0.25
        )
    else:
        print("could not find:", cat_path)