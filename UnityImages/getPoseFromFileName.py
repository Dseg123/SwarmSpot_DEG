import re

def parse_filename_to_pose(filename):
    # Split the filename by underscores
    parts = filename.split('_')

    # Function to convert string to float, handling scientific notation
    def parse_float(s):
        return float(s.replace(',', '.'))

    # Extracting position, rotation, and scale
    # Adjust the indices based on your filename structure
    position = [parse_float(parts[i]) for i in range(2, 5)]
    rotation = [parse_float(parts[i]) for i in range(6, 9)]
    scale = [parse_float(parts[i]) for i in range(10, 13)]

    return position, rotation, scale

# Example usage
# filename = "Camera_Pos_-2,09_12,29035_-36,14065_Rot_35,25964_3,56181_-1,306992E-07_Scale_1_1_1_img.png"

import os

directory = './Line/images/'
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        print(filename)
        position, rotation, scale = parse_filename_to_pose(filename)
        print("Position:", position)
        print("Rotation:", rotation)
        print("Scale:", scale)
    else:
        continue
# filename = "Camera_Pos_-1,389999_12,29035_-36,14065_Rot_35,3054_1,254644_3,269327E-08_Scale_1_1_1_img.png"
# position, rotation, scale = parse_filename_to_pose(filename)
# print("Position:", position)
# print("Rotation:", rotation)
# print("Scale:", scale)
