import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

from utils import apply_nms_to_predictions

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    for param in model.parameters():
        param.requires_grad = True
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    return model

def evaluate_map(model, data_loader, device, targets=None, conf_threshold=0.5):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format='xyxy').to(device)
    
    model.eval()
    with torch.no_grad():
        if isinstance(data_loader, list):
            predictions = model(data_loader)
            filtered_predictions = []
            for pred in predictions:
                boxes, labels, scores = apply_nms_to_predictions(
                    pred['boxes'],
                    pred['labels'],
                    pred['scores'],
                    score_threshold=conf_threshold
                )
                filtered_predictions.append({
                    'boxes': boxes,
                    'labels': labels,
                    'scores': scores
                })
            if targets:
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            metric.update(filtered_predictions, targets)
        else:
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images)
                
                filtered_predictions = []
                for pred in predictions:
                    boxes, labels, scores = apply_nms_to_predictions(
                        pred['boxes'],
                        pred['labels'],
                        pred['scores'],
                        score_threshold=conf_threshold
                    )
                    filtered_predictions.append({
                        'boxes': boxes,
                        'labels': labels,
                        'scores': scores
                    })
                
                metric.update(filtered_predictions, targets)
    
    metrics = metric.compute()
    return metrics['map_50'].item()

