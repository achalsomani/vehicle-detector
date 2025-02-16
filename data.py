import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

class VehicleDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, is_train=True):
        self.img_dir = img_dir
        self.is_train = is_train
        self.transform = transform or self.get_default_transforms(is_train)
        
        # Group annotations by image_id
        self.image_annotations = {}
        with open(label_file, 'r') as f:
            for line in f:
                values = line.strip().split()
                img_id = int(values[0])
                class_id = int(values[1])
                bbox = list(map(float, values[2:]))  # [cx, cy, w, h]
                
                # Convert to XYXY format
                x1 = bbox[0] - bbox[2]/2
                y1 = bbox[1] - bbox[3]/2
                x2 = bbox[0] + bbox[2]/2
                y2 = bbox[1] + bbox[3]/2
                
                if img_id not in self.image_annotations:
                    self.image_annotations[img_id] = {
                        'boxes': [],
                        'labels': []
                    }
                
                self.image_annotations[img_id]['boxes'].append([x1, y1, x2, y2])
                self.image_annotations[img_id]['labels'].append(class_id)
        
        # Convert to list for indexing
        self.image_ids = sorted(list(self.image_annotations.keys()))

    def set_overfit_dataset_size(self, overfit_dataset_size):
        self.image_ids = self.image_ids[:overfit_dataset_size]
        self.image_annotations = {k: self.image_annotations[k] for k in self.image_ids}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]  # Get the image_id from our list
        ann = self.image_annotations[img_id]  # Use image_id to get annotations
        img_name = f"{img_id:05d}.jpeg"  # Use actual image_id for filename
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.tensor(ann['labels'], dtype=torch.long)
        boxes = torch.clamp(boxes, min=0.0, max=1000.0)
        
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes.tolist(),
                class_labels=labels.tolist()
            )
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'])
            labels = torch.tensor(transformed['class_labels'])
            
            boxes = torch.clamp(boxes, min=0.0, max=1024.0)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        return image, target

    @staticmethod
    def get_default_transforms(train=True):
        if train:
            return A.Compose([
                A.Resize(1024, 1024),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(1024, 1024),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def get_dataloaders(train_img_dir, train_label_file, val_img_dir, val_label_file, 
                    batch_size=4, num_workers=4, overfit_dataset_size=None):
    train_dataset = VehicleDataset(train_img_dir, train_label_file, is_train=True)
    val_dataset = VehicleDataset(val_img_dir, val_label_file, is_train=False)
    
    if overfit_dataset_size is not None:
        train_dataset.set_overfit_dataset_size(overfit_dataset_size)
        val_dataset.set_overfit_dataset_size(overfit_dataset_size)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return torch.stack(images), targets 