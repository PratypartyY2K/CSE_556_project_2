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
                # Standard OBJ format uses 1-based indexing
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])
    return np.array(vertices), np.array(faces)


def main():
    # 1. Setup Paths
    points_path = os.path.join("colmap", "points3D.txt")
    inliers_path = os.path.join("output", "inlier_ids.npy")
    transform_path = os.path.join("output", "euclidean_transform.npz")
    asset_path = os.path.join("src", "assets", "icosahedron.txt")

    # 2. Load Required Data
    if not os.path.exists(transform_path):
        print("Error: euclidean_transform.npz not found in src/output/")
        return

    data = np.load(transform_path)
    R = data["R"]  # Rotation: Local -> Scene
    t = data["t"]  # Origin: Scene coordinates

    vertices, faces = load_icosahedron(asset_path)
    all_scene_points = load_colmap_points(points_path)
    inlier_indices = np.load(inliers_path)

    # 3. Align, Invert, and Scale
    # Find the 'bottom' vertices (those with the minimum Z coordinate)
    z_min = np.min(vertices[:, 2])
    bottom_v_idx = np.where(np.abs(vertices[:, 2] - z_min) < 1e-5)[0]
    bottom_center = np.mean(vertices[bottom_v_idx], axis=0)

    # Shift so bottom center is at (0,0,0)
    vertices_local = vertices - bottom_center

    # FLIP: Move into negative Z-direction
    vertices_local[:, 2] *= -1

    # SCALE: Adjust as needed for scene visibility
    scale_factor = 1
    vertices_local *= scale_factor

    # 4. Transform to Scene Coordinates
    # P_scene = R * P_local + t
    vertices_scene = (vertices_local @ R.T) + t

    # 5. Save the Scene-Transformed Points
    output_path = os.path.join("output", "icosahedron_scene_full.npz")
    np.savez(output_path, vertices=vertices_scene, faces=faces)
    print(f"Saved transformed object to {output_path}")

    # 6. Visualization (Local System View)
    # Map scene points back to local to verify flatness
    pts_local = (all_scene_points - t) @ R

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot Background Scene (Grey)
    ax.scatter(
        pts_local[:, 0], pts_local[:, 1], pts_local[:, 2], c="lightgrey", s=1, alpha=0.1
    )

    # Plot Dominant Plane (Red)
    ax.scatter(
        pts_local[inlier_indices, 0],
        pts_local[inlier_indices, 1],
        pts_local[inlier_indices, 2],
        c="red",
        s=2,
        alpha=0.3,
    )

    # --- COLORFUL ICOSAHEDRON ---
    # Create polygons for each face
    poly_faces = [vertices_local[face] for face in faces]

    # Generate a unique color for each of the 20 faces
    colors = plt.cm.viridis(np.linspace(0, 1, len(faces)))

    collection = Poly3DCollection(
        poly_faces, facecolors=colors, edgecolors="black", alpha=0.8
    )
    ax.add_collection3d(collection)

    # Plot Origin (Blue)
    ax.scatter([0], [0], [0], c="blue", s=200, marker="X", label="Local Origin")

    # Fix Aspect Ratio (Prevents squeezing)
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
