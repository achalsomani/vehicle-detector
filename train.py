import os
import torch
from tqdm import tqdm
from data import get_dataloaders
from model import get_model, evaluate_map
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, config, log_freq):
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
            writer.add_scalar('loss/train', current_loss, global_step)
            
            model.eval()
            with torch.no_grad():
                metrics = evaluate_map(model, images, device, targets, conf_threshold=config['conf_threshold'])
                # Overall metrics
                writer.add_scalar('metrics/train_mAP', metrics['map'], global_step)
                writer.add_scalar('metrics/train_precision', metrics['precision'], global_step)
                writer.add_scalar('metrics/train_recall', metrics['recall'], global_step)
                # Per-class metrics
                writer.add_scalar('class_metrics/train_precision_small', metrics['precision_small'], global_step)
                writer.add_scalar('class_metrics/train_precision_medium', metrics['precision_medium'], global_step)
                writer.add_scalar('class_metrics/train_precision_large', metrics['precision_large'], global_step)
                writer.add_scalar('class_metrics/train_recall_small', metrics['recall_small'], global_step)
                writer.add_scalar('class_metrics/train_recall_medium', metrics['recall_medium'], global_step)
                writer.add_scalar('class_metrics/train_recall_large', metrics['recall_large'], global_step)
            model.train()
    
    return total_loss / len(data_loader)

def compute_validation_loss(model, data_loader, device):
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
                
            # Sum up all the losses
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1
    
    return total_loss / num_batches

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
        'conf_threshold': 0.1,
        'num_epochs': 10,
        'device': 'cuda',
        'log_freq': 100,
        'overfit_dataset_size': 10
    }
    
    # Create a simple run name with timestamp
    run_name = datetime.now().strftime('run_%m%d_%H%M%S')
    if overfit:
        run_name += "_overfit"
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    
    # Log hyperparameters
    writer.add_text('config', str(config))
    
    # Create output directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        config['train_img_dir'],
        config['train_label_file'],
        config['val_img_dir'] if not overfit else config['train_img_dir'],  # Use train data for validation when overfitting
        config['val_label_file'] if not overfit else config['train_label_file'],  # Use train labels for validation when overfitting
        config['batch_size'],
        config['num_workers'],
        overfit_dataset_size=config['overfit_dataset_size'] if overfit else None
    )
    
    print("Train dataset size:", len(train_loader.dataset))
    print("Val dataset size:", len(val_loader.dataset))
    
    for images, targets in val_loader:
        print("Sample targets:", targets)
        break
    
    model = get_model(config['num_classes'])
    model = model.to(config['device'])
    
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
        val_loss = compute_validation_loss(model, val_loader, config['device'])
        metrics = evaluate_map(model, val_loader, config['device'], conf_threshold=config['conf_threshold'])
        
        # Log losses
        writer.add_scalar('loss/train_epoch', train_loss, epoch)
        writer.add_scalar('loss/val_epoch', val_loss, epoch)
        
        map_score = metrics['map']
        
        # Log validation metrics
        writer.add_scalar('metrics/val_mAP', metrics['map'], epoch)
        writer.add_scalar('metrics/val_precision', metrics['precision'], epoch)
        writer.add_scalar('metrics/val_recall', metrics['recall'], epoch)
        
        # Log per-class metrics
        writer.add_scalar('class_metrics/val_precision_small', metrics['precision_small'], epoch)
        writer.add_scalar('class_metrics/val_precision_medium', metrics['precision_medium'], epoch)
        writer.add_scalar('class_metrics/val_precision_large', metrics['precision_large'], epoch)
        writer.add_scalar('class_metrics/val_recall_small', metrics['recall_small'], epoch)
        writer.add_scalar('class_metrics/val_recall_medium', metrics['recall_medium'], epoch)
        writer.add_scalar('class_metrics/val_recall_large', metrics['recall_large'], epoch)

        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation mAP: {map_score:.4f}")
        print(f"Overall - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Save best model with same run name
        if map_score > best_map:
            best_map = map_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': map_score,
            }, f'checkpoints/{run_name}_best.pth')
    
    writer.close()

if __name__ == "__main__":
    main(overfit=True)