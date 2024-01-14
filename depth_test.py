import numpy as np
import time
import os
import sys
import torch
from tkinter import filedialog
import matplotlib.cm as cm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add path to ZoeDepth
sys.path.insert(0, "ZoeDepth")

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
from PIL import Image
import torch
import open3d as o3d

model_zoe_n = torch.hub.load("ZoeDepth", "ZoeD_N", source="local", pretrained=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

color_o3d = o3d.io.read_image("UnityImages/test_img.png")

image = Image.open("UnityImages/test_img.png").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)
depth_o3d = o3d.geometry.Image(depth_numpy)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# image2 = Image.open("UnityImages/test_img.png").convert("RGB")  # load
# depth_numpy2 = zoe.infer_pil(image2)

plt.imshow(depth_numpy)
plt.show()
