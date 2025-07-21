import os
import zipfile
import shutil

DATASET_ZIP = 'dataset/asl_alphabet_train.zip'  # Change if your zip has a different name
DATASET_DIR = 'dataset'

# 1. Extract zip file
with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
    zip_ref.extractall(DATASET_DIR)

# 2. Find the extracted root folder (e.g., 'asl_alphabet_train')
extracted_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
if len(extracted_folders) == 1:
    root_folder = os.path.join(DATASET_DIR, extracted_folders[0])
else:
    root_folder = DATASET_DIR

# 3. Move class folders up to dataset/
for folder in os.listdir(root_folder):
    src = os.path.join(root_folder, folder)
    dst = os.path.join(DATASET_DIR, folder)
    if os.path.isdir(src) and not os.path.exists(dst):
        shutil.move(src, dst)

# 4. Rename folders for consistency
rename_map = {'space': 'Space', 'delete': 'Delete', 'nothing': 'Nothing'}
for old, new in rename_map.items():
    old_path = os.path.join(DATASET_DIR, old)
    new_path = os.path.join(DATASET_DIR, new)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)

# 5. Remove the original extracted folder if empty
if root_folder != DATASET_DIR and os.path.exists(root_folder):
    try:
        os.rmdir(root_folder)
    except OSError:
        pass  # Not empty

# 6. Print summary
print('Dataset folders in', DATASET_DIR)
for folder in sorted(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, folder)
    if os.path.isdir(path):
        print('-', folder, ':', len(os.listdir(path)), 'images') 