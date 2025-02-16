import os
import torch
from tqdm import tqdm
from data import get_dataloaders
from model import get_model, evaluate_map
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, config, log_freq=100):
    model.train()
    total_loss = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        current_loss = losses.item()
        pbar.set_postfix({'loss': current_loss})
        
        global_step = epoch * len(data_loader) + batch_idx
        if batch_idx % log_freq == 0:
            writer.add_scalar('Loss/train', current_loss, global_step)
            
            if batch_idx % (log_freq * 5) == 0:
                model.eval()
                with torch.no_grad():
                    metrics = evaluate_map(model, images, device, targets, conf_threshold=config['conf_threshold'])
                    # Overall metrics
                    writer.add_scalar('Metrics/train_mAP', metrics['map'], global_step)
                    writer.add_scalar('Metrics/train_precision', metrics['precision'], global_step)
                    writer.add_scalar('Metrics/train_recall', metrics['recall'], global_step)
                    # Per-class metrics
                    writer.add_scalar('Metrics/train_precision_small', metrics['precision_small'], global_step)
                    writer.add_scalar('Metrics/train_precision_medium', metrics['precision_medium'], global_step)
                    writer.add_scalar('Metrics/train_precision_large', metrics['precision_large'], global_step)
                    writer.add_scalar('Metrics/train_recall_small', metrics['recall_small'], global_step)
                    writer.add_scalar('Metrics/train_recall_medium', metrics['recall_medium'], global_step)
                    writer.add_scalar('Metrics/train_recall_large', metrics['recall_large'], global_step)
                model.train()
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Need to pass targets to get loss during evaluation
            loss_dict = model(images, targets)
            
            # Handle both dictionary and list cases
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
            else:
                # Skip if no loss is computed
                continue
    
    return total_loss / len(data_loader)

def main(overfit=False):
    # Configuration
    config = {
        'train_img_dir': 'dataset/train/images',
        'train_label_file': 'dataset/train/labels.txt',
        'val_img_dir': 'dataset/val/images',
        'val_label_file': 'dataset/val/labels.txt',
        'num_classes': 3,  # This will become 4 with background
        'batch_size': 4,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'conf_threshold': 0.01,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log_freq': 100,
        'overfit_dataset_size': 10
    }
    
    # Initialize TensorBoard writer with unique run name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"vehicle_detection_{timestamp}"
    if overfit:
        run_name += "_overfit"
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    
    # Log hyperparameters
    writer.add_text('config', str(config))
    
    # Create output directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        config['train_img_dir'],
        config['train_label_file'],
        config['val_img_dir'],
        config['val_label_file'],
        config['batch_size'],
        config['num_workers'],
        overfit_dataset_size=config['overfit_dataset_size'] if overfit else None
    )
    
    # Initialize model
    model = get_model(config['num_classes'])
    model = model.to(config['device'])
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_map = 0
    for epoch in range(config['num_epochs']):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, 
            config['device'], epoch, writer, 
            config,
            config['log_freq']
        )
        
        # Evaluate
        val_loss = evaluate(model, val_loader, config['device'])
        metrics = evaluate_map(model, val_loader, config['device'], conf_threshold=config['conf_threshold'])
        
        # Log losses
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        
        map_score = metrics['map']
        
        # Log validation metrics
        writer.add_scalar('Metrics/val_mAP', metrics['map'], epoch)
        writer.add_scalar('Metrics/val_precision', metrics['precision'], epoch)
        writer.add_scalar('Metrics/val_recall', metrics['recall'], epoch)
        # Per-class validation metrics
        writer.add_scalar('Metrics/val_precision_small', metrics['precision_small'], epoch)
        writer.add_scalar('Metrics/val_precision_medium', metrics['precision_medium'], epoch)
        writer.add_scalar('Metrics/val_precision_large', metrics['precision_large'], epoch)
        writer.add_scalar('Metrics/val_recall_small', metrics['recall_small'], epoch)
        writer.add_scalar('Metrics/val_recall_medium', metrics['recall_medium'], epoch)
        writer.add_scalar('Metrics/val_recall_large', metrics['recall_large'], epoch)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation mAP: {map_score:.4f}")
        print(f"Overall - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print("Per-class metrics:")
        print(f"Small  - Precision: {metrics['precision_small']:.4f}, Recall: {metrics['recall_small']:.4f}")
        print(f"Medium - Precision: {metrics['precision_medium']:.4f}, Recall: {metrics['recall_medium']:.4f}")
        print(f"Large  - Precision: {metrics['precision_large']:.4f}, Recall: {metrics['recall_large']:.4f}")
        
        # Save best model
        if map_score > best_map:
            best_map = map_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': map_score,
            }, f'checkpoints/best_model.pth')
    
    writer.close()

if __name__ == "__main__":
    main(overfit=False)