import os
import torch
from tqdm import tqdm
from data import get_dataloaders
from model import get_model, evaluate_map
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, log_freq=100):
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
        
        # Simplified logging
        global_step = epoch * len(data_loader) + batch_idx
        if batch_idx % log_freq == 0:
            # Log only total loss
            writer.add_scalar('Loss/train', current_loss, global_step)
            
            # Calculate and log metrics during training (less frequently)
            if batch_idx % (log_freq * 5) == 0:
                model.eval()
                with torch.no_grad():
                    metrics = evaluate_map(model, images, device, targets)
                    # Overall metrics
                    writer.add_scalar('Metrics/train_mAP', metrics['map'], global_step)
                    writer.add_scalar('Metrics/train_precision', metrics['precision'], global_step)
                    writer.add_scalar('Metrics/train_recall', metrics['recall'], global_step)
                    # Per-class metrics
                    writer.add_scalar('Metrics/train_precision_car', metrics['precision_car'], global_step)
                    writer.add_scalar('Metrics/train_precision_mid', metrics['precision_mid'], global_step)
                    writer.add_scalar('Metrics/train_precision_large', metrics['precision_large'], global_step)
                    writer.add_scalar('Metrics/train_recall_car', metrics['recall_car'], global_step)
                    writer.add_scalar('Metrics/train_recall_mid', metrics['recall_mid'], global_step)
                    writer.add_scalar('Metrics/train_recall_large', metrics['recall_large'], global_step)
                model.train()
    
    return total_loss / len(data_loader)

def main(debug=False):
    # Configuration
    config = {
        'train_img_dir': 'dataset/train/images',
        'train_label_file': 'dataset/train/labels.txt',
        'val_img_dir': 'dataset/val/images',
        'val_label_file': 'dataset/val/labels.txt',
        'num_classes': 3,
        'batch_size': 4,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'num_epochs': 10,  
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log_freq': 100, 
        'debug_dataset_size': 10
    }
    
    # Initialize TensorBoard writer with unique run name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"vehicle_detection_{timestamp}"
    if debug:
        run_name += "_debug"
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
        debug_size=config['debug_dataset_size'] if debug else None
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
            config['log_freq']
        )
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        
        # Evaluate
        metrics = evaluate_map(model, val_loader, config['device'])
        map_score = metrics['map']
        
        # Log validation metrics
        writer.add_scalar('Metrics/val_mAP', metrics['map'], epoch)
        writer.add_scalar('Metrics/val_precision', metrics['precision'], epoch)
        writer.add_scalar('Metrics/val_recall', metrics['recall'], epoch)
        # Per-class validation metrics
        writer.add_scalar('Metrics/val_precision_car', metrics['precision_car'], epoch)
        writer.add_scalar('Metrics/val_precision_mid', metrics['precision_mid'], epoch)
        writer.add_scalar('Metrics/val_precision_large', metrics['precision_large'], epoch)
        writer.add_scalar('Metrics/val_recall_car', metrics['recall_car'], epoch)
        writer.add_scalar('Metrics/val_recall_mid', metrics['recall_mid'], epoch)
        writer.add_scalar('Metrics/val_recall_large', metrics['recall_large'], epoch)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation mAP: {map_score:.4f}")
        print(f"Overall - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print("Per-class metrics:")
        print(f"Car   - Precision: {metrics['precision_car']:.4f}, Recall: {metrics['recall_car']:.4f}")
        print(f"Mid   - Precision: {metrics['precision_mid']:.4f}, Recall: {metrics['recall_mid']:.4f}")
        print(f"Large - Precision: {metrics['precision_large']:.4f}, Recall: {metrics['recall_large']:.4f}")
        
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
    main(debug=False) 