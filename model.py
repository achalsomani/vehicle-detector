import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Freeze the entire backbone
    for param in model.backbone.body.parameters():
        param.requires_grad = False
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Optionally, verify frozen status
    for name, param in model.named_parameters():
        if 'backbone.body' in name:  # ResNet backbone
            print(f"Frozen - {name}: {param.requires_grad}")
        else:  # FPN and detection heads
            print(f"Trainable - {name}: {param.requires_grad}")
    
    return model

def evaluate_map(model, data_loader, device, targets=None):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(device)
    
    model.eval()
    with torch.no_grad():
        if isinstance(data_loader, list):
            predictions = model(data_loader)
            if targets:
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            metric.update(predictions, targets)
        else:
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images)
                metric.update(predictions, targets)
    
    metrics = metric.compute()
    
    # Safely get per-class metrics with more error handling
    try:
        if torch.is_tensor(metrics['map_per_class']):
            map_per_class = metrics['map_per_class'].tolist()
            # Pad with zeros if we have fewer than 3 classes
            map_per_class.extend([0.0] * (3 - len(map_per_class)))
        else:
            map_per_class = [0.0, 0.0, 0.0]
            
        if torch.is_tensor(metrics['mar_100_per_class']):
            mar_per_class = metrics['mar_100_per_class'].tolist()
            # Pad with zeros if we have fewer than 3 classes
            mar_per_class.extend([0.0] * (3 - len(mar_per_class)))
        else:
            mar_per_class = [0.0, 0.0, 0.0]
    except:
        map_per_class = [0.0, 0.0, 0.0]
        mar_per_class = [0.0, 0.0, 0.0]
    
    # Return overall and per-class metrics
    return {
        'map': metrics['map'].item() if torch.is_tensor(metrics['map']) else metrics['map'],
        'precision': metrics['map_50'].item() if torch.is_tensor(metrics['map_50']) else metrics['map_50'],
        'recall': metrics['mar_100'].item() if torch.is_tensor(metrics['mar_100']) else metrics['mar_100'],
        'precision_car': map_per_class[0],
        'precision_mid': map_per_class[1],
        'precision_large': map_per_class[2],
        'recall_car': mar_per_class[0],
        'recall_mid': mar_per_class[1],
        'recall_large': mar_per_class[2]
    } 