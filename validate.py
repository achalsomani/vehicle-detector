import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from data import VehicleDataset, collate_fn
from torch.utils.data import DataLoader
from model import evaluate_map
from test import load_checkpoint
from torchvision.ops import nms
from paths import val_img_dir, val_label_file
from utils import apply_nms_to_predictions

def evaluate_thresholds(model, val_loader, device, thresholds):
    results = []
    
    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        map = evaluate_map(
            model, 
            val_loader, 
            device, 
            conf_threshold=threshold
        )
        
        results.append({
            'threshold': threshold,
            'map': map,
        })
    
    return results


def visualize_image_predictions(image, predictions, save_path='prediction.png'):
    img_np = image.cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(img_np)
    
    colors = ['r', 'g', 'b']
    
    boxes, labels, scores = apply_nms_to_predictions(
        predictions['boxes'],
        predictions['labels'],
        predictions['scores']
    )
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.cpu().numpy()
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, color=colors[label-1], linewidth=2
        ))
        plt.text(x1, y1, f'Class {label.item()}: {score.item():.2f}', 
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def main():
    config = {
        'num_classes': 3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'val_img_dir': val_img_dir,
        'val_label_file': val_label_file,
        'subset_size': 10
    }
    
    # Load model (same as before)
    checkpoints_dir = 'checkpoints'
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('_210217_best.pth')]
    if not checkpoint_files:
        raise ValueError("No checkpoint files found!")
    
    latest_checkpoint = sorted(checkpoint_files)[-1]
    checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint and print stored mAP
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    if 'map' in checkpoint:
        print(f"\nBest mAP from checkpoint: {checkpoint['map']:.4f}")
    
    # Fixed: Added num_classes parameter
    model = load_checkpoint(checkpoint_path, config['num_classes'], config['device'])
    
    # Create output directory
    run_name = latest_checkpoint.replace('_best.pth', '')
    output_dir = f'validation/run_{run_name}_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create validation dataset
    val_dataset = VehicleDataset(
        config['val_img_dir'], 
        config['val_label_file'], 
        is_train=False
    )
    
    # Use only subset_size images
    val_dataset.set_overfit_dataset_size(config['subset_size'])
    print(f"Using {len(val_dataset)} validation images")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,  # Reduced workers for smaller dataset
        collate_fn=collate_fn
    )
    
    # Evaluate model at different thresholds
    thresholds = [0.1, 0.5]
    results = evaluate_thresholds(model, val_loader, config['device'], thresholds)
    
    # Save numerical results
    with open(os.path.join(output_dir, 'threshold_results.txt'), 'w') as f:
        f.write("Threshold | mAP \n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['threshold']:.1f} | {r['map']:.4f}\n")

    # Create output directory
    os.makedirs('predictions', exist_ok=True)
    
    # Get a few predictions
    model.eval()
    with torch.no_grad():
        for i in range(50):  # Look at first 5 images
            image, _ = val_dataset[i]
            predictions = model([image.to(config['device'])])[0]
            
            # Save predictions at different thresholds
            save_path = f'predictions/image_{i}.png'
            visualize_image_predictions(
                image, 
                predictions,
                save_path=save_path
            )
            print(f"Processed image {i}")

if __name__ == "__main__":
    main() 