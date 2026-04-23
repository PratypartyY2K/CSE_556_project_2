"""
ransac.py

Minimal, from-scratch RANSAC implementation to detect the dominant plane in a COLMAP
3D point cloud.

- Loads 3D points (and their COLMAP point IDs) from `colmap/points3D.txt`.
- Repeatedly samples 3 points to hypothesize a plane, computes the plane normal via a
  cross product, and counts inliers whose point-to-plane distance is below a threshold.
- Returns the best inlier set found after a fixed number of iterations (defaults:
  5000 iterations, distance threshold 0.05).

When run as a script:
- Reads `colmap/points3D.txt`
- Runs RANSAC on the coordinates
- Saves the *inlier indices* (into the loaded points array) to `output/inlier_ids.npy`
  for downstream visualization/processing.
"""

import numpy as np
import random
import os


def load_colmap_points(file_path):
    """Parses COLMAP points3D.txt to extract coordinates and IDs."""
    points = []
    ids = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            
            ids.append(int(parts[0]))
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(ids), np.array(points)


def fit_plane_ransac(points, iterations=5000, threshold=0.05):
    """Finds the dominant plane using RANSAC from scratch."""
    best_inlier_indices = []
    n_points = points.shape[0]

    for _ in range(iterations):
        
        idx = random.sample(range(n_points), 3)
        p1, p2, p3 = points[idx]

        
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if norm < 1e-6:
            continue  

        normal = normal / norm
        d = -np.dot(normal, p1)

        
        distances = np.abs(np.dot(points, normal) + d)
        inlier_indices = np.where(distances < threshold)[0]

        if len(inlier_indices) > len(best_inlier_indices):
            best_inlier_indices = inlier_indices

    return best_inlier_indices


if __name__ == "__main__":
    
    input_path = os.path.join("colmap", "points3D.txt")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    
    ids, pts = load_colmap_points(input_path)
    print(f"Loaded {len(pts)} points. Running RANSAC to find dominant plane")

    inlier_idx = fit_plane_ransac(pts)

    
    inlier_ids = ids[inlier_idx]
    save_path = os.path.join(output_dir, "inlier_ids.npy")
    np.save(save_path, inlier_idx)  

    print(f"Found {len(inlier_ids)} inliers.")
    print(f"Indices saved to {save_path}")
