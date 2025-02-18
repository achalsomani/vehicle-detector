import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from data import VehicleDataset, collate_fn
from torch.utils.data import DataLoader
from model import get_model, evaluate_map_with_details
from test import load_checkpoint

def evaluate_thresholds(model, val_loader, device, thresholds):
    results = []
    
    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        metrics = evaluate_map_with_details(
            model, 
            val_loader, 
            device, 
            conf_threshold=threshold
        )
        
        results.append({
            'threshold': threshold,
            'map': metrics['map'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'precision_small': metrics['precision_small'],
            'precision_medium': metrics['precision_medium'],
            'precision_large': metrics['precision_large']
        })
    
    return results

def plot_metrics(results, save_dir):
    thresholds = [r['threshold'] for r in results]
    
    # Plot mAP, Precision, Recall
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [r['map'] for r in results], 'b-', label='mAP')
    plt.plot(thresholds, [r['precision'] for r in results], 'g-', label='Precision')
    plt.plot(thresholds, [r['recall'] for r in results], 'r-', label='Recall')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Model Performance vs Confidence Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'threshold_metrics.png'))
    plt.close()
    
    # Plot per-class precision
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [r['precision_small'] for r in results], 'b-', label='Small')
    plt.plot(thresholds, [r['precision_medium'] for r in results], 'g-', label='Medium')
    plt.plot(thresholds, [r['precision_large'] for r in results], 'r-', label='Large')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title('Per-class Precision vs Confidence Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'threshold_precision_by_class.png'))
    plt.close()

def visualize_image_predictions(image, predictions, save_path='prediction.png'):
    """
    Draw predictions on a single image with just class and score
    """
    # Convert tensor image to numpy for visualization
    img_np = image.cpu().permute(1, 2, 0).numpy()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(img_np)
    
    # Colors for different classes
    colors = ['r', 'g', 'b']  # red for class 1, green for 2, blue for 3
    
    # Filter predictions by threshold
    mask = predictions['scores'] >= 0.1
    boxes = predictions['boxes'][mask]
    labels = predictions['labels'][mask]
    scores = predictions['scores'][mask]
    
    # Draw each prediction
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
        'num_classes': 3,  # 3 classes + background
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'val_img_dir': '/WAVE/projects/CSEN-342-Wi25/data/pr2/val/images',
        'val_label_file': '/WAVE/projects/CSEN-342-Wi25/data/pr2/val/labels.txt',
        'subset_size': 100  # Only use 50 images for quick testing
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
    thresholds = np.arange(0.1, 0.6, 0.2)
    results = evaluate_thresholds(model, val_loader, config['device'], thresholds)
    
    # Plot results
    plot_metrics(results, output_dir)
    
    # Save numerical results
    with open(os.path.join(output_dir, 'threshold_results.txt'), 'w') as f:
        f.write("Threshold | mAP | Precision | Recall | Small | Medium | Large\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['threshold']:.1f} | {r['map']:.4f} | {r['precision']:.4f} | "
                   f"{r['recall']:.4f} | {r['precision_small']:.4f} | "
                   f"{r['precision_medium']:.4f} | {r['precision_large']:.4f}\n")

    # Create output directory
    os.makedirs('predictions', exist_ok=True)
    
    # Get a few predictions
    model.eval()
    with torch.no_grad():
        for i in range(5):  # Look at first 5 images
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