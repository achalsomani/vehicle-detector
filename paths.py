


import os


if os.path.exists('/WAVE/projects/CSEN-342-Wi25/data/pr2/train/images'):
    train_img_dir = '/WAVE/projects/CSEN-342-Wi25/data/pr2/train/images'
    val_img_dir = '/WAVE/projects/CSEN-342-Wi25/data/pr2/val/images'
    train_label_file = '/WAVE/projects/CSEN-342-Wi25/data/pr2/train/labels.txt'
    val_label_file = '/WAVE/projects/CSEN-342-Wi25/data/pr2/val/labels.txt'
else:
    train_img_dir = 'dataset/train/images'
    val_img_dir = 'dataset/val/images'
    train_label_file = 'dataset/train/labels.txt'
    val_label_file = 'dataset/val/labels.txt'

test_img_dir = 'dataset/test/images'
test_label_file = 'dataset/test/labels.txt'