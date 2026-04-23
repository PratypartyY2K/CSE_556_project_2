# CSE 556 Project 2 — AR Overlay Pipeline

This repo contains a small pipeline to (1) extract frames, (2) detect a dominant plane in a COLMAP point cloud, (3) build a plane-aligned local coordinate system, (4) place a simple 3D mesh (icosahedron) into the scene, (5) render the mesh into each image, and (6) stitch the rendered frames into a final video.

## Environment setup

Run everything from the **repo root**.

## Project Structure

```text
.
|-- colmap/                  # COLMAP reconstruction files
|   |-- cameras.txt
|   |-- images.txt
|   `-- points3D.txt
|-- src/
|   |-- assets/
|   |   |-- images/          # Extracted video frames
|   |   |-- icosahedron.txt  # OBJ-style object geometry
|   |   `-- main_video.mp4
|   |-- ransac.py            # Dominant plane detection
|   |-- transform.py         # Local plane coordinate frame estimation
|   |-- visualize_scene.py   # Object placement and 3D visualization
|   |-- render.py            # Frame-by-frame AR overlay rendering
|   `-- make_video.py        # Rendered frame stitching
`-- output/                  # Generated transforms, frames, plots, and video
```

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Expected inputs / folders

- **COLMAP outputs**: `colmap/points3D.txt`, `colmap/cameras.txt`, `colmap/images.txt`
- **Video + images + mesh asset** (already in this repo): `src/assets/`
  - `src/assets/main_video.mp4`
  - `src/assets/images/` (input images used by COLMAP and for overlay rendering)
  - `src/assets/icosahedron.txt`
- **Generated outputs** (written by scripts): `output/`

## Execution order (recommended)

### 0) (Optional) Extract frames from a video

This saves **every 10th frame** from `src/assets/main_video.mp4` into `src/assets/images/`.

```bash
python src/__init__.py
```

If you already have your `src/assets/images/` folder and COLMAP was run on it, you can skip this step.

### 1) Find dominant plane in the COLMAP point cloud (RANSAC)

Reads `colmap/points3D.txt` and writes `output/inlier_ids.npy`.

```bash
python src/ransac.py
```

### 2) (Optional) Visualize the dominant plane result

Uses `output/inlier_ids.npy` to plot inliers vs outliers.

```bash
python src/projection.py
```

### 3) Compute plane-aligned Euclidean transform

Computes a local coordinate system aligned to the plane and saves:
- `output/euclidean_transform.npz` (contains `R`, `t`)

```bash
python src/transform.py
```

### 4) Place the icosahedron into the scene and save mesh in scene coordinates

Produces:
- `output/icosahedron_scene_full.npz` (scene-space `vertices`, `faces`)

Choose one of the following:

- **Just compute + save**:

```bash
python src/object_handling.py
```

- **Compute + visualize in global scene coordinates** (also saves the same `.npz`):

```bash
python src/visualize_scene.py
```

### 5) Render the mesh into each image (overlay)

Reads:
- `output/icosahedron_scene_full.npz`
- `colmap/cameras.txt`, `colmap/images.txt`
- `src/assets/images/<image_name>` for each image listed in `images.txt`

Writes:
- `output/updated_frames/<image_name>.png`

```bash
python src/render.py
```

### 6) Stitch overlays into a final MP4

Reads `output/updated_frames/` and writes `output/final_render.mp4`.

```bash
python src/make_video.py
```

## One-shot command list

```bash
source .venv/bin/activate
pip install -r requirements.txt

python src/ransac.py
python src/transform.py
python src/object_handling.py    # or: python src/visualize_scene.py
python src/render.py
python src/make_video.py
```

