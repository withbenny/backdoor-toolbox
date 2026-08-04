import os, shutil

val_dir = 'data/tinyimagenet/val'
with open(os.path.join(val_dir, 'val_annotations.txt')) as f:
    for line in f:
        img, cls = line.split('\t')[:2]
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        shutil.move(
            os.path.join(val_dir, 'images', img),
            os.path.join(val_dir, cls, img)
        )