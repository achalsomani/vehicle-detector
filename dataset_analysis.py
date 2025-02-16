import os
from collections import Counter

def read_labels(file_path):
    """Read and parse label file."""
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            class_id = int(float(line.strip().split()[1]))
            boxes.append(class_id)
    return boxes

def main():
    # Read train and validation labels
    train_labels = read_labels('dataset/train/labels.txt')
    val_labels = read_labels('dataset/val/labels.txt')

    # Get class distributions
    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)
    
    print("\nClass Distribution:")
    print("Class 1 (Car/SUV/Van/Small Truck):")
    print(f"  Train: {train_dist[1]}")
    print(f"  Val: {val_dist[1]}")
    
    print("\nClass 2 (Medium Truck - Delivery):")
    print(f"  Train: {train_dist[2]}")
    print(f"  Val: {val_dist[2]}")
    
    print("\nClass 3 (Large Truck - 18-wheeler/Bus):")
    print(f"  Train: {train_dist[3]}")
    print(f"  Val: {val_dist[3]}")

if __name__ == "__main__":
    main()
