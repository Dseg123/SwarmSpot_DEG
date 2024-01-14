import numpy as np
import time
import os
import sys
import torch
from tkinter import filedialog
import matplotlib.cm as cm

np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add path to ZoeDepth
sys.path.insert(0, "ZoeDepth")
sys.path.insert(0, "icp")

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
from PIL import Image
import torch
import open3d as o3d

# Units are relative and scaled to 0.0 to 1.0
def visualize_depth_image(depth_map, path='depth_map_visualization.png'):
        # Normalize the depth map for visualization
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    # Plotting
    plt.imshow(depth_map_normalized, cmap=cm.jet)
    plt.colorbar()  # Adds a color bar to show the depth scale
    plt.title("Depth Map Visualization")
    plt.savefig(path)  # Save the figure as a PNG
    # plt.show()  # Show the plot in a window

def show_3D_depth(depth_map, path='3D_depth.png'):
    # Create meshgrid for plotting
    x = np.arange(depth_map.shape[1])
    y = np.arange(depth_map.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, depth_map, cmap='viridis', edgecolor='none')

    # Add color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Depth')

    fig.savefig(path)  # Save the figure as a PNG

    # Show the plot
    # plt.show()


# torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh 

# Z's, and camera pos are unknown, 
# How to make correspondences between points 

# unknowns are camera pos and Z's 
# Non-linear quadratic optimization

########## LINE ##########
# left_im_path = "/Users/jeffrey/Coding/hackathons/hackathon_schmidt_01_13/UnityImages/Line/images/Camera_Pos_-0,289999_12,29035_-36,14065_Rot_35,28864_357,6237_0_Scale_1_1_1_img.png"
left_im_path = "/Users/jeffrey/Coding/hackathons/hackathon_schmidt_01_13/UnityImages/Line/images/Camera_Pos_-0,289999_12,29035_-36,14065_Rot_35,28864_357,6237_0_Scale_1_1_1_img.png"
right_im_path = "/Users/jeffrey/Coding/hackathons/hackathon_schmidt_01_13/UnityImages/Line/images/Camera_Pos_-2,09_12,29035_-36,14065_Rot_35,25964_3,56181_-1,306992E-07_Scale_1_1_1_img.png"
# right_im_path = "/Users/jeffrey/Coding/hackathons/hackathon_schmidt_01_13/UnityImages/Line/images/Camera_Pos_-0,789999_12,29035_-36,14065_Rot_35,3097_359,2736_4,904251E-08_Scale_1_1_1_img.png"
# left_im_path = "images/sample_left.png"
# right_im_path = "images/sample_right.png"

left_im = Image.open(left_im_path).convert("RGB")  # load
right_im = Image.open(right_im_path).convert("RGB")  # load



# intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
# 1500 0 360 0 1000 240 0 0 1
# print(intrinsic.intrinsic_matrix)

intrinsic = o3d.camera.PinholeCameraIntrinsic(720, 480, 1500, 1000, 360, 240)

print(intrinsic.intrinsic_matrix)
# x= 0 


left_image = Image.open(left_im_path).convert("RGB")
left_color_o3d = o3d.io.read_image(left_im_path) # uint8

right_image = Image.open(right_im_path).convert("RGB")
right_color_o3d = o3d.io.read_image(right_im_path) # uint8

# depth_pil = zoe.infer_pil(image, output_type='pil')
# depth_pil.save('sample_depth.png')



if not os.path.exists(left_im_path[:-4] + '_zoe.npy') or not os.path.exists(right_im_path[:-4] + '_zoe.npy'):
    start = time.time()
    model_zoe_n = torch.hub.load("ZoeDepth", "ZoeD_N", source="local", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)
    left_depth_numpy = zoe.infer_pil(left_image)  # as numpy
    right_depth_numpy = zoe.infer_pil(right_image)  # as numpy
    print(time.time() - start)
    np.save(left_im_path[:-4] + '_zoe.npy', np.asarray(left_depth_numpy))
    np.save(right_im_path[:-4] + '_zoe.npy', np.asarray(right_depth_numpy))
else:
    left_depth_numpy = np.load(left_im_path[:-4] + '_zoe.npy')
    right_depth_numpy = np.load(right_im_path[:-4] + '_zoe.npy')

visualize_depth_image(left_depth_numpy, left_im_path[:-4]+"_depth.png")
visualize_depth_image(right_depth_numpy, right_im_path[:-4]+"_depth.png")

show_3D_depth(left_depth_numpy, left_im_path[:-4]+"_3Ddepth.png")
show_3D_depth(right_depth_numpy, right_im_path[:-4]+"_3Ddepth.png")

x = 0
left_depth_numpy = left_depth_numpy.astype(np.float32)
right_depth_numpy = right_depth_numpy.astype(np.float32)

left_depth_o3d = o3d.geometry.Image(left_depth_numpy) # Convert to open3d image
right_depth_o3d = o3d.geometry.Image(right_depth_numpy) # Convert to open3d image


# [fx, 0, cx], [0, fy, cy], [0, 0, 1]
# intrinsic = o3d.camera.PinholeCameraIntrinsic(733, 450, 50, 50, 2, 11)

# left_pcd = o3d.geometry.PointCloud.create_from_depth_image(
    # left_depth_o3d, intrinsic)

# right_pcd = o3d.geometry.PointCloud.create_from_depth_image(
    # left_depth_o3d, intrinsic)

# np.asarray(left_depth_o3d)[0, 0] = 1.62
# np.asarray(left_rgbd_image.depth)[0, 0] = 0.00162

left_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    left_color_o3d, left_depth_o3d, depth_scale=1000.0)

right_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    right_color_o3d, right_depth_o3d, depth_scale=1000.0)

# # print(rgbd_image)
left_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    left_rgbd_image, intrinsic)


right_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    right_rgbd_image, intrinsic)

combined_pcd = left_pcd + right_pcd


# print(pcd)
o3d.visualization.draw_geometries([combined_pcd])


from icp import icp

transform, distances, iters = icp(np.asarray(left_pcd.points), np.asarray(right_pcd.points), max_iterations=20, tolerance=0.00001)

print(distances, iters)

print(np.asarray(right_pcd.points).shape)

ones = np.ones((len(np.asarray(right_pcd.points)), 1))
right_points_homo = np.hstack((np.asarray(right_pcd.points), ones))

transformed_right_pcd_homo_points = (transform @ right_points_homo.T).T
transformed_right_pcd_points = transformed_right_pcd_homo_points[:, :3] / transformed_right_pcd_homo_points[:, 3, np.newaxis]

transformed_right_pcd = o3d.geometry.PointCloud()
transformed_right_pcd.points = o3d.utility.Vector3dVector(transformed_right_pcd_points)

combined_align_pcd = left_pcd + transformed_right_pcd

o3d.visualization.draw_geometries([combined_align_pcd])

x = 0
# o3d.io.write_image("sample_depth.png", depth_o3d)


# import matplotlib.pyplot as plt
# plt.imshow(depth_pil)
# plt.show()
# depth_pil.save("sample_depth.jpg")