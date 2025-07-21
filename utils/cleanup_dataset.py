import os
import shutil

DATASET_DIR = 'dataset'
# List of valid class folder names
valid_classes = [chr(i) for i in range(65, 91)] + ['Space', 'Nothing', 'Delete']

removed = []
kept = []

for folder in os.listdir(DATASET_DIR):
    path = os.path.join(DATASET_DIR, folder)
    if os.path.isdir(path):
        if folder not in valid_classes:
            shutil.rmtree(path)
            removed.append(folder)
        else:
            kept.append(folder)

print('Removed folders:', removed)
print('Kept class folders:', sorted(kept)) 