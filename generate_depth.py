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

def save_depth(img_path, depth_path):
    image = Image.open(img_path)  # load
    depth_np = zoe.infer_pil(image)
    np.save(depth_path, depth_np)

if __name__ == "__main__":
    col_path1 = '/home/dylaneg/Documents/Programming/NYCHackathon/hackathon_schmidt_thrun_jan_2024/UnityImages/Line/images/Camera_Pos_-0,3899994_12,29035_-36,14065_Rot_35,29464_357,9535_6,537785E-08_Scale_1_1_1_img.png'
    col_path2 = '/home/dylaneg/Documents/Programming/NYCHackathon/hackathon_schmidt_thrun_jan_2024/UnityImages/Line/images/Camera_Pos_3,610001_12,29035_-36,14065_Rot_34,39165_345,0897_5,173158E-07_Scale_1_1_1_img.png'
    save_depth(col_path1, 'UnityImages/depth_img.npy')
    save_depth(col_path2, 'UnityImages/depth2_img.npy')