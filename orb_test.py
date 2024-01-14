import cv2
import numpy as np
from PIL import Image
from scipy.linalg import svd


def estimate_initial_transform(source_points, target_points):
    # Center the points
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    # Singular Value Decomposition
    H = centered_source.T @ centered_target
    U, _, Vt = svd(H)

    # Rotation matrix
    R = Vt.T @ U.T

    # Translation vector
    t = centroid_target - R @ centroid_source

    return R, t


# Assuming you have two grayscale images: img1 and img2
img1 = np.asarray(Image.open('UnityImages/test_img.png'))
print(img1)
img2 = np.asarray(Image.open('UnityImages/test2_img.png'))
print(img2)

# Step 1: Detect ORB keypoints and descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Step 2: Match descriptors using a matcher (e.g., BFMatcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Step 3: Filter matches based on a distance threshold
distance_threshold = 50  # Adjust as needed
good_matches = [match for match in matches if match.distance < distance_threshold]

# Step 4: Extract corresponding 3D points from the matched keypoints
corresponding_points_img1 = np.array([keypoints1[match.queryIdx].pt for match in good_matches])
corresponding_points_img2 = np.array([keypoints2[match.trainIdx].pt for match in good_matches])

print(corresponding_points_img1)
print(corresponding_points_img2)

# Convert 2D points to homogeneous 3D points by adding depth information (from depth images)
# Use camera calibration parameters if available
depth_img1 = np.load('UnityImages/depth_img.npy')  # Depth image corresponding to img1
depth_img2 = np.load('UnityImages/depth2_img.npy')  # Depth image corresponding to img2

# # Assuming (u, v) are the pixel coordinates of the keypoints
corresponding_points_3d_img1 = np.hstack((corresponding_points_img1, depth_img1[corresponding_points_img1[:, 1].astype(int), corresponding_points_img1[:, 0].astype(int)].reshape(-1, 1)))
corresponding_points_3d_img2 = np.hstack((corresponding_points_img2, depth_img2[corresponding_points_img2[:, 1].astype(int), corresponding_points_img2[:, 0].astype(int)].reshape(-1, 1)))

print(corresponding_points_3d_img1)
print(corresponding_points_3d_img2)
# # Optionally, you might want to further filter or refine the correspondences

# # Now, use the corresponding 3D points for initial transformation estimation
initial_rotation, initial_translation = estimate_initial_transform(corresponding_points_3d_img1, corresponding_points_3d_img2)

# Continue with ICP or other alignment methods
print(initial_rotation)
print(initial_translation)
