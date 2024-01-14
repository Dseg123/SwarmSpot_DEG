import numpy as np
from scipy.linalg import svd

# Assuming you have two depth images as numpy arrays: depth_image1 and depth_image2

# # Step 1: Convert depth images to 3D point clouds
# def depth_to_point_cloud(depth_image, fx, fy, cx, cy):
#     rows, cols = depth_image.shape
#     y, x = np.mgrid[:rows, :cols]
#     points = np.stack([(x - cx) * depth_image / fx, (y - cy) * depth_image / fy, depth_image], axis=-1)
#     return points

def downsample(points, factor):
    indices = np.random.choice(len(points), size=int(len(points) * factor), replace=False)
    return points[indices]


def get_essential(points1, points2):
    points1 = downsample(points1, 0.1)
    points2 = downsample(points2, 0.1)
    print(points1.shape)
    print(points2.shape)

    # Replace fx, fy, cx, cy with the corresponding camera intrinsic parameters
    fx = fy = cx = cy = 1.0  # Replace with actual values

    # Step 2: Reshape points to 2D arrays
    points1 = points1.reshape((-1, 3))
    points2 = points2.reshape((-1, 3))

    # Step 3: Normalize points (optional, but recommended for numerical stability)
    mean1 = np.mean(points1, axis=0)
    mean2 = np.mean(points2, axis=0)
    points1_normalized = points1 - mean1
    points2_normalized = points2 - mean2

    # Step 4: Compute the Essential Matrix
    A = np.zeros((len(points1), 9))
    A[:, 0:3] = points1_normalized
    A[:, 3:6] = points2_normalized
    A[:, 6:9] = np.ones((len(points1), 3))

    print(A.shape)
    _, _, V = svd(A)

    # Extract the last column of V to obtain the Essential Matrix
    E_vector = V[-1, :]
    Essential_matrix = E_vector.reshape((3, 3))

    return Essential_matrix

if __name__ == "__main__":
    points = np.load('sample_points.npy')
    points1 = points
    points2 = points
    E = (get_essential(points1, points2))
    U, S, Vt = np.linalg.svd(E)
    S[-1] = 0
    Essential_matrix_normalized = U @ np.diag(S) @ Vt
    print(Essential_matrix_normalized)

    # Assuming you have computed the Essential Matrix (Essential_matrix_normalized)

    # Step 1: Perform Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(Essential_matrix_normalized)

    # Ensure that the determinant of U and Vt is positive (to make them proper rotation matrices)
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[2, :] *= -1

    # Step 2: Create rotation and translation matrices
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # Skew-symmetric matrix

    # Two possible solutions for rotation and translation
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    # Step 3: Create 4 possible transformation matrices
    T1 = np.hstack((R1, t.reshape(-1, 1)))
    T2 = np.hstack((R1, -t.reshape(-1, 1)))
    T3 = np.hstack((R2, t.reshape(-1, 1)))
    T4 = np.hstack((R2, -t.reshape(-1, 1)))

    # Note: You would need additional information or constraints to select the correct solution

    # Print or use the rotation matrix and translation vector
    print("Rotation Matrix 1:")
    print(R2)
    print("Translation Vector:")
    print(t)

