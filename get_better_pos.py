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

intrinsic = o3d.camera.PinholeCameraIntrinsic(720, 480, 1500, 1000, 360, 240)
def transform_points(points_3d, initial_transform_matrix):
    # Step 1: Convert to homogeneous coordinates
    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Step 2: Apply the transformation matrix
    transformed_points_homogeneous = initial_transform_matrix @ points_homogeneous.T
    transformed_points_homogeneous = transformed_points_homogeneous[:3, :].T  # Remove the homogeneous coordinate

    # Step 3: Convert back to 3D coordinates
    transformed_points_3d = transformed_points_homogeneous[:, :3]

    return transformed_points_3d





from evaluation import evaluate



def get_T(col1, depth1, col2, depth2):
    init_pose = get_init_pos2(col1, depth1, col2, depth2)

    color1_o3d = o3d.geometry.Image(col1)
    color2_o3d = o3d.geometry.Image(col2)

    depth1_o3d = o3d.geometry.Image(depth1)
    depth2_o3d = o3d.geometry.Image(depth2)

    rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color1_o3d, depth1_o3d)
    rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color2_o3d, depth2_o3d)

    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd1,
        intrinsic)
    print(pcd1)
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd2,
        intrinsic)
    print(pcd2)

    pcd1.scale(1000, np.array([0, 0, 0]))
    pcd2.scale(1000, np.array([0, 0, 0]))

    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)



    T, distances, i = icp(points1, points2, init_pose)
    
    return points1, points2, T


if __name__ == '__main__':
    col_path1 = '/home/dylaneg/Documents/Programming/NYCHackathon/hackathon_schmidt_thrun_jan_2024/UnityImages/Line/images/Camera_Pos_-1,889999_12,29035_-36,14065_Rot_35,27717_2,90347_6,536374E-08_Scale_1_1_1_img.png'
    col_path2 = '/home/dylaneg/Documents/Programming/NYCHackathon/hackathon_schmidt_thrun_jan_2024/UnityImages/Line/images/Camera_Pos_-4,389999_12,29035_-36,14065_Rot_34,81016_11,02346_0_Scale_1_1_1_img.png'
    depth_path1 = 'UnityImages/depth_img.npy'
    depth_path2 = 'UnityImages/depth2_img.npy'

    col1 = np.asarray(Image.open(col_path1))
    col2 = np.asarray(Image.open(col_path2))
    depth1 = np.load(depth_path1)
    depth2 = np.load(depth_path2)

    points1, points2, T = get_T(col1, depth1, col2, depth2)
    print(points1, points2)

    print(T)
    points3 = transform_points(points2, np.linalg.inv(T))
    print("Initial Score:", evaluate(points1, points2))
    print("Score:", evaluate(points1, points3))