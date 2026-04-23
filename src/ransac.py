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
            # ID is col 0, X, Y, Z are cols 1, 2, 3
            ids.append(int(parts[0]))
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(ids), np.array(points)


def fit_plane_ransac(points, iterations=5000, threshold=0.05):
    """Finds the dominant plane using RANSAC from scratch."""
    best_inlier_indices = []
    n_points = points.shape[0]

    for _ in range(iterations):
        # Sample 3 unique points
        idx = random.sample(range(n_points), 3)
        p1, p2, p3 = points[idx]

        # Calculate plane normal using cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if norm < 1e-6:
            continue  # Skip collinear points

        normal = normal / norm
        d = -np.dot(normal, p1)

        # Vectorized distance calculation: |ax + by + cz + d|
        distances = np.abs(np.dot(points, normal) + d)
        inlier_indices = np.where(distances < threshold)[0]

        if len(inlier_indices) > len(best_inlier_indices):
            best_inlier_indices = inlier_indices

    return best_inlier_indices


if __name__ == "__main__":
    # Paths relative to root
    input_path = os.path.join("colmap", "points3D.txt")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Run routine
    ids, pts = load_colmap_points(input_path)
    print(f"Loaded {len(pts)} points. Running RANSAC...")

    inlier_idx = fit_plane_ransac(pts)

    # Save the inlier IDs for the plotter to use
    inlier_ids = ids[inlier_idx]
    save_path = os.path.join(output_dir, "inlier_ids.npy")
    np.save(save_path, inlier_idx)  # Saving indices for easier plotting later

    print(f"Found {len(inlier_ids)} inliers.")
    print(f"Indices saved to {save_path}")
