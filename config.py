import os


if os.path.exists('/WAVE/projects/CSEN-342-Wi25/data/pr2/test/images'):
    batch_size = 16
    num_workers = 16
else:
    batch_size = 4
    num_workers = 4
