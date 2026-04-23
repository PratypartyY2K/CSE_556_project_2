"""
object_handling.py
Loads a simple 3D mesh (an icosahedron) and places it into a reconstructed COLMAP scene
using a precomputed Euclidean transform.
Workflow:
- Reads COLMAP point cloud coordinates from `colmap/points3D.txt`.
- Loads an icosahedron mesh (OBJ-like `v`/`f` lines) from `src/assets/icosahedron.txt`.
- Loads `output/euclidean_transform.npz` containing:
  - R: rotation mapping local/object coordinates -> scene coordinates
  - t: translation/origin in scene coordinates
- Recenters the icosahedron so its “bottom” (min-Z vertices) is at the local origin,
  flips it into negative Z, optionally scales it, then transforms it into scene space:
  P_scene = P_local @ R.T + t
- Saves the transformed mesh to `output/icosahedron_scene_full.npz` (vertices, faces).
- Visualizes the scene in the *local* coordinate system by mapping COLMAP points back
  via (P_scene - t) @ R, highlighting inlier plane points from `output/inlier_ids.npy`,
  and rendering the colored icosahedron for sanity-checking alignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


def load_colmap_points(file_path):
    """Parses COLMAP points3D.txt to extract coordinates."""
    points = []
    if not os.path.exists(file_path):
        return np.array([])
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(points)


def load_icosahedron(file_path):
    """Parses vertices and faces from the icosahedron.txt file."""
    vertices = []
    faces = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:]])
            elif line.startswith("f "):
                
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])
    return np.array(vertices), np.array(faces)


def main():
    
    points_path = os.path.join("colmap", "points3D.txt")
    inliers_path = os.path.join("output", "inlier_ids.npy")
    transform_path = os.path.join("output", "euclidean_transform.npz")
    asset_path = os.path.join("src", "assets", "icosahedron.txt")

    
    if not os.path.exists(transform_path):
        print("Error: euclidean_transform.npz not found in src/output/")
        return

    data = np.load(transform_path)
    R = data["R"]  
    t = data["t"]  

    vertices, faces = load_icosahedron(asset_path)
    all_scene_points = load_colmap_points(points_path)
    inlier_indices = np.load(inliers_path)

    
    
    z_min = np.min(vertices[:, 2])
    bottom_v_idx = np.where(np.abs(vertices[:, 2] - z_min) < 1e-5)[0]
    bottom_center = np.mean(vertices[bottom_v_idx], axis=0)

    
    vertices_local = vertices - bottom_center

    
    vertices_local[:, 2] *= -1

    
    scale_factor = 1
    vertices_local *= scale_factor

    
    
    vertices_scene = (vertices_local @ R.T) + t

    
    output_path = os.path.join("output", "icosahedron_scene_full.npz")
    np.savez(output_path, vertices=vertices_scene, faces=faces)
    print(f"Saved transformed object to {output_path}")

    
    
    pts_local = (all_scene_points - t) @ R

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    
    ax.scatter(
        pts_local[:, 0], pts_local[:, 1], pts_local[:, 2], c="lightgrey", s=1, alpha=0.1
    )

    
    ax.scatter(
        pts_local[inlier_indices, 0],
        pts_local[inlier_indices, 1],
        pts_local[inlier_indices, 2],
        c="red",
        s=2,
        alpha=0.3,
    )

    
    
    poly_faces = [vertices_local[face] for face in faces]

    
    colors = plt.cm.viridis(np.linspace(0, 1, len(faces)))

    collection = Poly3DCollection(
        poly_faces, facecolors=colors, edgecolors="black", alpha=0.8
    )
    ax.add_collection3d(collection)

    
    ax.scatter([0], [0], [0], c="blue", s=200, marker="X", label="Local Origin")

    
    max_range = (
        np.array(
            [
                pts_local[:, 0].max() - pts_local[:, 0].min(),
                pts_local[:, 1].max() - pts_local[:, 1].min(),
                pts_local[:, 2].max() - pts_local[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x, mid_y, mid_z = np.mean(pts_local, axis=0)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title("Colored Icosahedron in Negative Z Space")
    plt.show()


if __name__ == "__main__":
    main()
