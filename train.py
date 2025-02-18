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
            
            # Evaluate mAP on current batch
            model.eval()
            with torch.no_grad():
                map_score = evaluate_map(model, images, device, targets, conf_threshold=config['conf_threshold'])
                writer.add_scalar('metrics/train_mAP', map_score, global_step)
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
        'train_img_dir': '/WAVE/projects/CSEN-342-Wi25/data/pr2/train/images',
        'train_label_file': '/WAVE/projects/CSEN-342-Wi25/data/pr2/train/labels.txt',
        'val_img_dir': '/WAVE/projects/CSEN-342-Wi25/data/pr2/val/images',
        'val_label_file': '/WAVE/projects/CSEN-342-Wi25/data/pr2/val/labels.txt',
        'num_classes': 3,
        'batch_size':16,
        'num_workers': 16,
        'backbone_lr': 5e-5,
        'classifier_lr': 1e-4,
        'conf_threshold': 0.1,
        'num_epochs': 20,
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
    writer.add_text('config', str(config))
    
    # Create output directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        config['train_img_dir'],
        config['train_label_file'],
        config['val_img_dir'] if not overfit else config['train_img_dir'],
        config['val_label_file'] if not overfit else config['train_label_file'],
        config['batch_size'],
        config['num_workers'],
        overfit_dataset_size=config['overfit_dataset_size'] if overfit else None
    )
    
    model = get_model(config['num_classes'])
    model = model.to(config['device'])
    
    # Separate backbone and classifier parameters
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if "box_predictor" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    # Create optimizer with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['backbone_lr']},
        {'params': classifier_params, 'lr': config['classifier_lr']}
    ])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=2, verbose=True
    )
    
    # Training loop
    best_map = 0
    for epoch in range(config['num_epochs']):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, 
            config['device'], epoch, writer, 
            config, config['log_freq']
        )
        
        # Evaluate
        val_loss = compute_validation_loss(model, val_loader, config['device'])
        val_map = evaluate_map(model, val_loader, config['device'], conf_threshold=config['conf_threshold'])
        
        # Log metrics
        writer.add_scalar('loss/train_epoch', train_loss, epoch)
        writer.add_scalar('loss/val_epoch', val_loss, epoch)
        writer.add_scalar('metrics/val_mAP', val_map, epoch)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation mAP: {val_map:.4f}")
        
        # Save best model
        if val_map > best_map:
            best_map = val_map
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'map': val_map,
            }, f'checkpoints/{run_name}_best.pth')
    
    writer.close()

if __name__ == "__main__":
    main(overfit=False)