"""
transform.py

Computes a Euclidean change-of-basis that defines a local coordinate system aligned to
the dominant plane in a COLMAP reconstruction, then visualizes and saves that transform.

- Loads the COLMAP point cloud from `colmap/points3D.txt`.
- Loads dominant-plane inlier indices from `output/inlier_ids.npy` (from the RANSAC step).
- Sets the local origin `t` as the centroid of the inlier points.
- Uses SVD/PCA on centered inliers to estimate an orthonormal basis:
  - u, v span the plane
  - w is the plane normal (flipped if needed to keep a right-handed frame)
- Constructs rotation matrix R = [u v w] and maps scene points into the local frame via:
  P_local = (P_scene - t) @ R
- Plots the transformed point cloud (outliers grey, plane inliers red, origin blue) for
  sanity checking.
- Saves `output/euclidean_transform.npz` containing `R` and `t` for downstream placement/rendering.
"""

import numpy as np
import os
import matplotlib.pyplot as plt


def load_colmap_points(file_path):
    points = []
    
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(points)


def compute_and_plot_transformation():
    
    data_path = os.path.join( "colmap", "points3D.txt")
    inlier_path = os.path.join("output", "inlier_ids.npy")

    if not os.path.exists(inlier_path):
        print(f"Error: Could not find {inlier_path}. Ensure RANSAC has run.")
        return

    
    all_points = load_colmap_points(data_path)
    inlier_indices = np.load(inlier_path)
    inlier_pts = all_points[inlier_indices]

    
    t = np.mean(inlier_pts, axis=0)

    
    centered_pts = inlier_pts - t
    _, _, vh = np.linalg.svd(centered_pts)

    u, v, w = vh[0], vh[1], vh[2]

    
    if np.dot(np.cross(u, v), w) < 0:
        w = -w

    
    R = np.column_stack((u, v, w))

    
    
    points_local = (all_points - t) @ R

    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    
    outlier_mask = np.ones(len(all_points), dtype=bool)
    outlier_mask[inlier_indices] = False
    ax.scatter(
        points_local[outlier_mask, 0],
        points_local[outlier_mask, 1],
        points_local[outlier_mask, 2],
        c="lightgrey",
        s=1,
        alpha=0.1,
    )

    
    ax.scatter(
        points_local[inlier_indices, 0],
        points_local[inlier_indices, 1],
        points_local[inlier_indices, 2],
        c="red",
        s=5,
        alpha=0.5,
        label="Dominant Plane (z=0)",
    )

    
    
    ax.scatter([0], [0], [0], c="blue", s=200, marker="X", label="Local Origin (t)")

    ax.set_xlabel("Local X")
    ax.set_ylabel("Local Y")
    ax.set_zlabel("Local Z")
    ax.set_title("Local Coordinate System with Origin Highlighted")

    
    ax.set_zlim([-5, 5])
    ax.legend()
    plt.show()

    
    np.savez(os.path.join("output", "euclidean_transform.npz"), R=R, t=t)


if __name__ == "__main__":
    compute_and_plot_transformation()
