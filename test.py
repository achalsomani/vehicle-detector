import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from data import VehicleTestDataset, test_collate_fn
from torch.utils.data import DataLoader
from model import get_model
from PIL import Image, ImageDraw
from paths import test_img_dir
from config import batch_size, num_workers


def load_checkpoint(checkpoint_path, num_classes, device):
    """Load model from checkpoint"""
    model = get_model(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def visualize_predictions(model, dataset, device, save_dir, num_samples=50):
    """Generate and save visualization of model predictions"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define colors for different classes
    colors = ['red', 'green', 'blue']
    class_names = ['small', 'medium', 'large']
    
    for i in range(num_samples):
        plt.figure(figsize=(12, 8))
        
        # Get sample
        image, target, original_image = dataset[i]
        
        # Get model predictions
        with torch.no_grad():
            prediction = model([image.to(device)])[0]
        
        original_image = original_image.resize(image.shape[1:3])

        draw = ImageDraw.Draw(original_image)
        
        # Draw predicted boxes
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            x1, y1, x2, y2 = box.cpu().numpy()
            label = label.cpu().item()
            score = score.cpu().item()
            
            # Draw box and label
            draw.rectangle([x1, y1, x2, y2], outline=colors[label-1], width=3)
            draw.text((x1, y1-10), f'{class_names[label-1]} {score:.2f}', 
                        fill=colors[label-1])
        
        plt.imshow(original_image)
        plt.axis('off')
        plt.title(f'Predictions - Sample {i+1}')
        plt.savefig(os.path.join(save_dir, f'prediction_{i+1}.png'))
        plt.close()

def generate_predictions_file(model, dataset, device, output_path):
    """Generate predictions file in the specified format"""
    model.eval()
    
    with open(output_path, 'w') as f:
        for idx in tqdm(range(len(dataset)), desc="Generating predictions"):
            image, target, _ = dataset[idx]
            image = image.to(device)
            
            with torch.no_grad():
                prediction = model([image])[0]
            
            image_id = target['image_id'].item()
            
            # Filter predictions by confidence threshold
            mask = prediction['scores'] > 0.1
            boxes = prediction['boxes'][mask]
            labels = prediction['labels'][mask]
            scores = prediction['scores'][mask]
            
            # Convert predictions to required format
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.cpu().numpy()
                # Convert to center format
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                # Write in required format
                f.write(f"{image_id} {label.item()} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score.item():.6f}\n")

def main():
    # Configuration
    config = {
        'num_classes': 3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'test_img_dir': test_img_dir,
        'batch_size': batch_size,
        'num_workers': num_workers
    }
    
    # Create output directory
    os.makedirs('test_output', exist_ok=True)
    
    # Find the latest checkpoint
    checkpoints_dir = 'checkpoints'
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('_210217_best.pth')]
    if not checkpoint_files:
        raise ValueError("No checkpoint files found!")
    
    latest_checkpoint = sorted(checkpoint_files)[-1]
    checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = load_checkpoint(checkpoint_path, config['num_classes'], config['device'])
    
    # Create test dataset
    test_dataset = VehicleTestDataset(config['test_img_dir'])
    
    # Generate visualizations
    visualize_predictions(model, test_dataset, config['device'], 'test_output')
    
    # Generate predictions file
    generate_predictions_file(model, test_dataset, config['device'], 'test_output/predictions.txt')

if __name__ == "__main__":
    main()