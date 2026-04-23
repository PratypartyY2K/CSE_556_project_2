"""
render.py

Renders a placed 3D mesh (e.g., an icosahedron) into each COLMAP input image by
projecting its vertices with the recovered camera intrinsics/extrinsics, then saving
overlay frames.

Key pieces:
- `load_camera_params(colmap_path)` parses:
  - `colmap/cameras.txt` to build the camera intrinsic matrix K (supports common pinhole variants)
  - `colmap/images.txt` to load per-image extrinsics (quaternion + translation) into {R, t}
- `project_points(points_3d, R, t, K)` applies a pinhole projection:
  P_cam = R * P_world + t, then pixel = K * (P_cam / z), keeping only points with z > 0

Main script behavior:
- Loads the scene-space mesh from `output/icosahedron_scene_full.npz` (vertices, faces).
- For each image listed in `images.txt`, loads the corresponding file from `src/assets/images/`.
- Computes an approximate painter’s ordering by sorting faces by average camera-space depth.
- Projects each triangular face and draws it as a semi-transparent colored polygon overlay.
- Saves the rendered overlays to `output/updated_frames/<image_name>.png`.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
import os


def quaternion_to_R(q):
    """Converts COLMAP quaternion (qw, qx, qy, qz) to 3x3 Rotation Matrix."""
    qw, qx, qy, qz = q
    return np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qw * qy,
                2 * qy * qz + 2 * qw * qx,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ]
    )


def load_camera_params(colmap_path):
    """Parses cameras.txt (intrinsics) and images.txt (extrinsics)."""
    K = None
    cam_file = os.path.join(colmap_path, "cameras.txt")
    if not os.path.exists(cam_file):
        raise FileNotFoundError(f"Could not find {cam_file}")

    with open(cam_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            model = parts[1]
            params = list(map(float, parts[4:]))

            
            if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                f_val, cx, cy = params[0], params[1], params[2]
                K = np.array([[f_val, 0, cx], [0, f_val, cy], [0, 0, 1]])
            elif model in ["PINHOLE", "RADIAL"]:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            else:
                print(
                    f"Warning: Unknown camera model '{model}'. Using first 3 params as f, cx, cy."
                )
                f_val, cx, cy = params[0], params[1], params[2]
                K = np.array([[f_val, 0, cx], [0, f_val, cy], [0, 0, 1]])
            break

    images_metadata = {}
    img_file = os.path.join(colmap_path, "images.txt")
    with open(img_file, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("#") or not line:
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 10:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                img_name = parts[9]
                images_metadata[img_name] = {
                    "R": quaternion_to_R([qw, qx, qy, qz]),
                    "t": np.array([tx, ty, tz]),
                }
                i += 2  
            else:
                i += 1
    return K, images_metadata


def project_points(points_3d, R, t, K):
    """Manually projects 3D world points to 2D pixels using pinhole model."""
    
    p_cam = (R @ points_3d.T).T + t
    depths = p_cam[:, 2]

    
    valid_mask = depths > 0
    if not np.any(valid_mask):
        return np.zeros((0, 2)), depths, valid_mask

    
    p_norm = p_cam[valid_mask] / p_cam[valid_mask, 2:3]

    
    p_pixel = (K @ p_norm.T).T
    return p_pixel[:, :2], depths, valid_mask


def main():
    colmap_path = "colmap"
    image_dir = "src/assets/images"
    object_path = os.path.join("output", "icosahedron_scene_full.npz")

    if not os.path.exists(object_path):
        print(f"Error: {object_path} not found. Run your placement script first.")
        return

    K, images_metadata = load_camera_params(colmap_path)
    if K is None:
        print("Error: Failed to load camera intrinsics from cameras.txt")
        return

    obj_data = np.load(object_path)
    vertices_world = obj_data["vertices"]
    faces = obj_data["faces"]

    
    cmap = plt.colormaps["viridis"]
    face_colors = [cmap(i / len(faces)) for i in range(len(faces))]

    
    for img_name in sorted(images_metadata.keys()):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        print(f"Rendering {img_name}...")
        img = mpimg.imread(img_path)
        R, t = images_metadata[img_name]["R"], images_metadata[img_name]["t"]

        
        v_cam = (R @ vertices_world.T).T + t
        face_depths = [np.mean(v_cam[f, 2]) for f in faces]
        sorted_indices = np.argsort(face_depths)[::-1]  

        plt.figure(figsize=(10, 8))
        plt.imshow(img)

        for idx in sorted_indices:
            face = faces[idx]
            
            p_pixel, _, valid = project_points(vertices_world[face], R, t, K)

            
            if len(p_pixel) == 3:
                poly = Polygon(
                    p_pixel,
                    closed=True,
                    facecolor=face_colors[idx],
                    edgecolor="black",
                    alpha=0.8,
                )
                plt.gca().add_patch(poly)

        plt.title(f"Overlay: {img_name}")
        plt.axis("off")
        plt.savefig(os.path.join("output","updated_frames", f"{img_name}.png"), bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
