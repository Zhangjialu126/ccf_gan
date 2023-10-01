import random
import os
import shutil

def random_move_dir(source_dir=None, target_dir=None):
    images_dirs = []
    for root, dirs, files in os.walk(source_dir):
        for sub in dirs:
            images_dirs.append(sub)
    print(len(images_dirs))

    n_classes = 200
    sample = random.sample(images_dirs, n_classes)
    print(len(sample))
    for i in sample:
        source_sub_dir = os.path.join(source_dir, i)
        target_sub_dir = os.path.join(target_dir, i)
        if os.path.exists(target_sub_dir):
            shutil.rmtree(target_sub_dir, ignore_errors=True)
        shutil.copytree(source_sub_dir, target_sub_dir)

if __name__ == '__main__':
    source_dir = r"./data/VGG-Face2"
    target_dir = r"./data/VGG-Face1000"
    random_move_dir(source_dir, target_dir)