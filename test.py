import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from data import VehicleDataset, collate_fn
from torch.utils.data import DataLoader
from model import get_model
import seaborn as sns
from PIL import Image, ImageDraw
import torchvision.transforms as T

def load_checkpoint(checkpoint_path, num_classes, device):
    """Load model from checkpoint"""
    model = get_model(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def visualize_predictions(model, dataset, device, save_dir, num_samples=3):
    """Generate and save visualization of model predictions"""
    # Define colors for different classes
    colors = ['red', 'green', 'blue']
    class_names = ['small', 'medium', 'large']
    
    for i in range(num_samples):
        plt.figure(figsize=(10, 10))
        
        # Get samples 1,2,3 instead of random
        idx = i  # This will get samples 0,1,2
        image, _ = dataset[idx]
        
        # Get model predictions
        with torch.no_grad():
            prediction = model([image.to(device)])[0]
        
        # Denormalize the image
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = image * std[:, None, None] + mean[:, None, None]
        
        # Convert tensor image back to PIL for drawing
        image_np = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        draw = ImageDraw.Draw(image_pil)
        
        # Draw predicted boxes
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score > 0.5:  # Only show predictions with confidence > 0.1
                x1, y1, x2, y2 = box.cpu().numpy()
                label = label.cpu().item()
                draw.rectangle([x1, y1, x2, y2], outline=colors[label-1], width=2)
                draw.text((x1, y1-10), f'{class_names[label-1]} {score:.2f}', fill=colors[label-1])
        
        plt.imshow(image_pil)
        plt.axis('off')
        plt.title(f'Predictions - Sample {i+1}')
        plt.savefig(os.path.join(save_dir, f'prediction_{i+1}.png'))
        plt.close()

def plot_confidence_distribution(all_confidences, all_classes, save_dir):
    """Plot confidence score distribution for each class"""
    plt.figure(figsize=(10, 6))
    
    class_names = ['small', 'medium', 'large']
    for i, class_name in enumerate(class_names, 1):
        class_conf = [conf for conf, cls in zip(all_confidences, all_classes) if cls == i]
        if class_conf:
            sns.kdeplot(data=class_conf, label=class_name)
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Score Distribution by Class')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'))
    plt.close()

def generate_predictions_file(model, dataset, device, output_path):
    """Generate predictions file in the specified format"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    
    with open(output_path, 'w') as f:
        for images, targets in tqdm(loader, desc="Generating predictions"):
            image = images[0].to(device)
            
            with torch.no_grad():
                prediction = model([image])[0]
            
            image_id = targets[0]['image_id'].item()
            
            # Convert predictions to required format
            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if score > 0.5:  # Only save predictions with confidence > 0.5
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
        'train_img_dir': 'dataset/train/images',
        'train_label_file': 'dataset/train/labels.txt',
    }
    
    # Find the latest checkpoint
    checkpoints_dir = 'checkpoints'
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('_best.pth')]
    if not checkpoint_files:
        raise ValueError("No checkpoint files found!")
    
    latest_checkpoint = sorted(checkpoint_files)[-1]
    checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
    
    # Create output directory
    run_name = latest_checkpoint.replace('_best.pth', '')
    output_dir = f'test/run_{run_name}_model'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_checkpoint(checkpoint_path, config['num_classes'], config['device'])
    
    # Create dataset
    dataset = VehicleDataset(config['train_img_dir'], config['train_label_file'], is_train=False)
    
    # Generate visualizations first
    visualize_predictions(model, dataset, config['device'], output_dir)
    
    # Generate predictions file
    generate_predictions_file(model, dataset, config['device'], 
                            os.path.join(output_dir, 'labels.txt'))

if __name__ == "__main__":
    main()
