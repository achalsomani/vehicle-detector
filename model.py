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

def evaluate_map(model, data_loader, device):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision()
    
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            # Convert predictions and targets to the format expected by MeanAveragePrecision
            metric.update(predictions, targets)
    
    return metric.compute() 