import os
import shutil

def sort_images(source_folder, img_folder, depth_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith("_img.png"):
            shutil.move(os.path.join(source_folder, filename), os.path.join(img_folder, filename))
        elif filename.endswith("_depth.png"):
            shutil.move(os.path.join(source_folder, filename), os.path.join(depth_folder, filename))

# Example usage
source_folder = '.'
img_folder = './images'
depth_folder = './depth'

# Ensure target folders exist
os.makedirs(img_folder, exist_ok=True)
os.makedirs(depth_folder, exist_ok=True)

sort_images(source_folder, img_folder, depth_folder)

