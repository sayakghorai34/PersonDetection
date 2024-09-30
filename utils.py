import torch
from torchvision.datasets import CocoDetection
import json
import os

def coco_dataset():
    dataset = CocoDetection(root='path_to_coco_images', annFile='path_to_coco_annotations.json')
    num_classes = 2
    return dataset, num_classes

def ochuman_dataset():
    dataset = CocoDetection(root='OCHuman/images', annFile='OCHuman/ochuman_coco_format_val_range_0.00_1.00.json')
    num_classes = 2
    return dataset, num_classes

def save_pseudo_labels(outputs, img_file, output_folder):
    annotations = []
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score > 0.5:
            xmin, ymin, xmax, ymax = box
            annotations.append({
                'bbox': [xmin.item(), ymin.item(), (xmax - xmin).item(), (ymax - ymin).item()],
                'label': label.item(),
                'score': score.item()
            })
    
    annotation_file = os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}.json")
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f)