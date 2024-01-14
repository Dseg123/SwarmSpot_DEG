import numpy as np
import time
import os
import sys
import torch
from tkinter import filedialog
import matplotlib.cm as cm

import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from init_pos import get_init_pos, get_init_pos2

sys.path.append('icp')
from icp import icp

from get_better_pos import get_T, transform_points
from icp import nearest_neighbor

# gets depth at x, y in col1
def depth_at_point(col1, col2, depth1, depth2, x, y):
    points1, points2, T = get_T(col1, depth1, col2, depth2)
    p = np.array([x, y])
    xy_1 = points1[:, :2]
    best = 1000000
    best_ind = 0
    for i in range(len(xy_1)):
        vec = (xy_1[i, :] - p)
        dist = np.sqrt(np.dot(vec, vec))
        if dist < best:
            best = dist
            best_ind = i
    
    depth1 = points1[best_ind, 2]

    points3 = transform_points(points2, np.linalg.inv(T))

    distances, indices = nearest_neighbor(points1, points3)

    depth2 = points3[indices[best_ind], 2]

    print("Transformation:\n", T)
    print("Depth from first image:", depth1)
    print("Depth from second image:", depth2)
    print("Averaged Depth:", (depth1 + depth2)/2)

    return (depth1 + depth2)/2

if __name__ == "__main__":
    col_path1 = '/home/dylaneg/Documents/Programming/NYCHackathon/hackathon_schmidt_thrun_jan_2024/UnityImages/Line/images/Camera_Pos_-0,3899994_12,29035_-36,14065_Rot_35,29464_357,9535_6,537785E-08_Scale_1_1_1_img.png'
    col_path2 = '/home/dylaneg/Documents/Programming/NYCHackathon/hackathon_schmidt_thrun_jan_2024/UnityImages/Line/images/Camera_Pos_3,610001_12,29035_-36,14065_Rot_34,39165_345,0897_5,173158E-07_Scale_1_1_1_img.png'
    depth_path1 = 'UnityImages/depth_img.npy'
    depth_path2 = 'UnityImages/depth2_img.npy'

    x = 0
    y = 0
    col1 = np.asarray(Image.open(col_path1))
    col2 = np.asarray(Image.open(col_path2))
    depth1 = np.load(depth_path1)
    depth2 = np.load(depth_path2)
    (depth_at_point(col1, col2, depth1, depth2, x, y))





