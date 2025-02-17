import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

def get_model(num_classes):
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    for param in model.parameters():
        param.requires_grad = True
    
    # Replace the classifier with num_classes + 1 (for background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    return model

def evaluate_map(model, data_loader, device, targets=None, conf_threshold=0.5):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(device)
    
    model.eval()
    with torch.no_grad():
        if isinstance(data_loader, list):
            predictions = model(data_loader)
            predictions = [{
                'boxes': pred['boxes'][pred['scores'] > conf_threshold],
                'labels': pred['labels'][pred['scores'] > conf_threshold],
                'scores': pred['scores'][pred['scores'] > conf_threshold]
            } for pred in predictions]
            if targets:
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            metric.update(predictions, targets)
        else:
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images)
                
                # Apply confidence threshold filtering to predictions
                filtered_predictions = [{
                    'boxes': pred['boxes'][pred['scores'] > conf_threshold],
                    'labels': pred['labels'][pred['scores'] > conf_threshold],
                    'scores': pred['scores'][pred['scores'] > conf_threshold]
                } for pred in predictions]
                
                metric.update(filtered_predictions, targets)
    
    metrics = metric.compute()
    
    # Initialize result with default values
    result = {
        'map': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'precision_small': 0.0,
        'precision_medium': 0.0,
        'precision_large': 0.0,
        'recall_small': 0.0,
        'recall_medium': 0.0,
        'recall_large': 0.0
    }
    
    # Update with actual values if they exist
    if torch.is_tensor(metrics['map']):
        result['map'] = metrics['map'].item()
    if torch.is_tensor(metrics['map_50']):
        result['precision'] = metrics['map_50'].item()
    if torch.is_tensor(metrics['mar_100']):
        result['recall'] = metrics['mar_100'].item()
        
    # Handle per-class metrics
    if 'map_per_class' in metrics and torch.is_tensor(metrics['map_per_class']):
        per_class_map = metrics['map_per_class'].cpu().numpy()
        if isinstance(per_class_map, np.ndarray):
            if per_class_map.ndim > 0:  # Check if it's not a scalar
                if per_class_map.size > 0: result['precision_small'] = max(0.0, float(per_class_map[0]))
                if per_class_map.size > 1: result['precision_medium'] = max(0.0, float(per_class_map[1]))
                if per_class_map.size > 2: result['precision_large'] = max(0.0, float(per_class_map[2]))
            else:
                result['precision_small'] = max(0.0, float(per_class_map))
        
    if 'mar_100_per_class' in metrics and torch.is_tensor(metrics['mar_100_per_class']):
        per_class_mar = metrics['mar_100_per_class'].cpu().numpy()
        if isinstance(per_class_mar, np.ndarray):
            if per_class_mar.ndim > 0:
                if per_class_mar.size > 0: result['recall_small'] = max(0.0, float(per_class_mar[0]))
                if per_class_mar.size > 1: result['recall_medium'] = max(0.0, float(per_class_mar[1]))
                if per_class_mar.size > 2: result['recall_large'] = max(0.0, float(per_class_mar[2]))
            else:
                result['recall_small'] = max(0.0, float(per_class_mar))
    
    return result 