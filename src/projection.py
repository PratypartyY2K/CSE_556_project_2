"""
projection.py

Simple visualization utility for inspecting COLMAP’s reconstructed 3D point cloud and a
RANSAC-detected dominant plane.

- Loads 3D point coordinates from `colmap/points3D.txt` (skips comments/blank lines).
- Loads inlier indices for the dominant plane from `output/inlier_ids.npy`
  (expected to be produced by `ransac_lib.py`).
- Plots the point cloud in 3D with Matplotlib:
  - outliers in light grey
  - plane inliers in red
- Opens an interactive 3D figure with a fixed viewing angle for quick sanity checks.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def visualize():
    
    data_path = os.path.join("colmap", "points3D.txt")
    inlier_path = os.path.join("output", "inlier_ids.npy")

    if not os.path.exists(inlier_path):
        print("Error: Run ransac_lib.py first to generate inlier data.")
        return

    
    points = []
    with open(data_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    points = np.array(points)

    
    inlier_indices = np.load(inlier_path)

    
    outlier_mask = np.ones(len(points), dtype=bool)
    outlier_mask[inlier_indices] = False

    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    
    ax.scatter(
        points[outlier_mask, 0],
        points[outlier_mask, 1],
        points[outlier_mask, 2],
        c="lightgrey",
        s=1,
        alpha=0.3,
        label="Scene Points",
    )

    
    ax.scatter(
        points[inlier_indices, 0],
        points[inlier_indices, 1],
        points[inlier_indices, 2],
        c="red",
        s=5,
        alpha=0.8,
        label="Dominant Plane",
    )

    ax.set_title("COLMAP 3D Point Cloud - Dominant Plane Detection")
    ax.legend()

    
    ax.view_init(elev=20, azim=45)

    print("Displaying plot. Close window to finish.")
    plt.show()


if __name__ == "__main__":
    visualize()
