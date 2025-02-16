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
    metric = MeanAveragePrecision().to(device)  # Move metric to device
    
    model.eval()
    with torch.no_grad():
        if isinstance(data_loader, list):  # For single batch evaluation
            predictions = model(data_loader)
            # Move targets to device if provided
            if targets:
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            metric.update(predictions, targets)
        else:  # For full dataset evaluation
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                # Move targets to device
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                predictions = model(images)
                metric.update(predictions, targets)
    
    return metric.compute() 