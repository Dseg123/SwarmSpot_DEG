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
from init_pos import get_init_pos

def transform_points(points_3d, initial_transform_matrix):
    # Step 1: Convert to homogeneous coordinates
    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Step 2: Apply the transformation matrix
    transformed_points_homogeneous = initial_transform_matrix @ points_homogeneous.T
    transformed_points_homogeneous = transformed_points_homogeneous[:3, :].T  # Remove the homogeneous coordinate

    # Step 3: Convert back to 3D coordinates
    transformed_points_3d = transformed_points_homogeneous[:, :3]

    return transformed_points_3d


col_path1 = '/home/dylaneg/Documents/Programming/NYCHackathon/hackathon_schmidt_thrun_jan_2024/UnityImages/Line/images/Camera_Pos_-1,889999_12,29035_-36,14065_Rot_35,27717_2,90347_6,536374E-08_Scale_1_1_1_img.png'
col_path2 = '/home/dylaneg/Documents/Programming/NYCHackathon/hackathon_schmidt_thrun_jan_2024/UnityImages/Line/images/Camera_Pos_-4,389999_12,29035_-36,14065_Rot_34,81016_11,02346_0_Scale_1_1_1_img.png'
depth_path1 = 'UnityImages/depth_img.npy'
depth_path2 = 'UnityImages/depth2_img.npy'

init_pose = get_init_pos(col_path1, depth_path1, col_path2, depth_path2)

# intrinsic = np.array([[1500, 0, 360], [0, 1000, 240], [0, 0, 1]])
intrinsic = o3d.camera.PinholeCameraIntrinsic(720, 480, 1500, 1000, 360, 240)

depth1 = np.load(depth_path1)
depth2 = np.load(depth_path2)

color1_o3d = o3d.io.read_image(col_path1)
color2_o3d = o3d.io.read_image(col_path2)

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
print(np.asarray(pcd1.points))
print(np.asarray(pcd2.points))

pcd1.scale(1000, np.array([0, 0, 0]))
pcd2.scale(1000, np.array([0, 0, 0]))
print("PCDS")
print(np.asarray(pcd1.points))
print(np.asarray(pcd2.points))
print(transform_points(np.asarray(pcd2.points), init_pose))

# print(transform_points(np.asarray(pcd2.points), init_pose))
# # print(o3d.camera.PinholeCameraIntrinsic(
# #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# print(np.asarray(pcd1.points))
# print(np.asarray(pcd2.points))

# print("Initial alignment:", init_pose)
# threshold = 0.02
# trans_init = init_pose
# evaluation = o3d.pipelines.registration.evaluate_registration(
#     pcd2, pcd1, threshold, trans_init)
# # print(evaluation.transformation)
# # print(init_pose)
# print(evaluation)

from evaluation import evaluate

points1 = np.asarray(pcd1.points)
points2 = np.asarray(pcd2.points)
print(points1, points2)

print(evaluate(points1, points2))

points3 = transform_points(points2, np.linalg.inv(init_pose))
print(evaluate(points1, points3))

sys.path.append('icp')
from icp import icp

T, distances, i = icp(points1, points2, init_pose)
print(init_pose)
print(T)

points4 = transform_points(points2, np.linalg.inv(T))
print(evaluate(points1, points4))

