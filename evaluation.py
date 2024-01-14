import numpy as np
import time
import os
import sys
import torch
from tkinter import filedialog
import matplotlib.cm as cm

sys.path.append('icp/')
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from icp import nearest_neighbor

def evaluate(points1, points2, threshold = 0.01):
    distances, indices = nearest_neighbor(points1, points2)
    return (distances < threshold).sum() / len(distances)


