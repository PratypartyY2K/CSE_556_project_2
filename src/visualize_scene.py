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
            # X, Y, Z are columns 1, 2, 3
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
                # OBJ indices are 1-based
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])
    return np.array(vertices), np.array(faces)


def main():
    # Define Paths
    points_path = os.path.join( "colmap", "points3D.txt")
    inliers_path = os.path.join("output", "inlier_ids.npy")
    transform_path = os.path.join("output", "euclidean_transform.npz")
    asset_path = os.path.join("src", "assets", "icosahedron.txt")

    # Load Data
    all_points = load_colmap_points(points_path)
    inlier_indices = np.load(inliers_path)
    trans_data = np.load(transform_path)
    R = trans_data["R"]  # This was defined as [u, v, w] as columns
    t = trans_data["t"]  # The scene centroid

    vertices, faces = load_icosahedron(asset_path)

    # Create Local Icosahedron (Centered, Flipped, and Scaled)
    # Identify the bottom face (min z)
    z_min = np.min(vertices[:, 2])
    bottom_v_idx = np.where(np.abs(vertices[:, 2] - z_min) < 1e-5)[0]
    bottom_center = np.mean(vertices[bottom_v_idx], axis=0)

    # Local Transformation (Origin at bottom center, negative z direction)
    vertices_local = vertices - bottom_center
    vertices_local[:, 2] *= -1  # Flip to negative z direction

    scale_factor = 5.0  # Adjust based on scene scale
    vertices_local *= scale_factor

    # Convert Local Points to Scene X, Y, Z
    # Using the change of basis: P_scene = P_local * R^T + t
    vertices_scene = (vertices_local @ R.T) + t

    # Visualization in Scene Coordinates
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot Scene Points (Global X,Y,Z)
    outlier_mask = np.ones(len(all_points), dtype=bool)
    outlier_mask[inlier_indices] = False
    ax.scatter(
        all_points[outlier_mask, 0],
        all_points[outlier_mask, 1],
        all_points[outlier_mask, 2],
        c="lightgrey",
        s=1,
        alpha=0.1,
        label="Other Scene Points",
    )

    # Highlight Plane Inliers
    ax.scatter(
        all_points[inlier_indices, 0],
        all_points[inlier_indices, 1],
        all_points[inlier_indices, 2],
        c="red",
        s=2,
        alpha=0.3,
        label="Dominant Plane",
    )

    # Plot the Transformed Icosahedron
    poly_faces = [vertices_scene[face] for face in faces]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(faces)))  # Diff colors for faces

    collection = Poly3DCollection(
        poly_faces, facecolors=colors, edgecolors="black", alpha=0.8
    )
    ax.add_collection3d(collection)

    # Plot Scene Origin (Local t)
    ax.scatter(
        [t[0]],
        [t[1]],
        [t[2]],
        c="blue",
        s=100,
        marker="X",
        label="Plane Center (Origin)",
    )

    # Final Plot Settings
    ax.set_xlabel("Global X")
    ax.set_ylabel("Global Y")
    ax.set_zlabel("Global Z")
    ax.set_title("Icosahedron Transformed into Global Scene Coordinates")

    # Set equal aspect ratio for global view
    max_range = (
        np.array(
            [
                all_points[:, 0].max() - all_points[:, 0].min(),
                all_points[:, 1].max() - all_points[:, 1].min(),
                all_points[:, 2].max() - all_points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x, mid_y, mid_z = np.mean(all_points, axis=0)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.legend()
    plt.show()

    # Save the scene-coordinates object
    np.savez(
        os.path.join("output", "icosahedron_scene_full.npz"),
        vertices=vertices_scene,
        faces=faces,
    )


if __name__ == "__main__":
    main()
