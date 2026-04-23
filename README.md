# CSE 586 / CV2 Project 2: AR Object Rendering

This project reconstructs a scene with COLMAP outputs, detects a dominant plane in
the 3D point cloud, places an icosahedron on that plane, projects the object into
the source video frames, and stitches the rendered frames into a final augmented
reality video.

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

## Requirements

The scripts use Python 3 with:

- `numpy`
- `matplotlib`
- `opencv-python`

Install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib opencv-python
```

## Inputs

The pipeline expects these files to exist:

- `colmap/points3D.txt`: sparse 3D scene points from COLMAP
- `colmap/cameras.txt`: camera intrinsics
- `colmap/images.txt`: camera poses and image names
- `src/assets/images/`: extracted source frames
- `src/assets/icosahedron.txt`: object mesh to insert into the scene

## Running the Pipeline

Run commands from the repository root.

1. Detect the dominant plane:

```bash
python3 src/ransac.py
```

This writes `output/inlier_ids.npy`.

2. Estimate the plane-aligned coordinate frame:

```bash
python3 src/transform.py
```

This writes `output/euclidean_transform.npz` and displays a 3D visualization.

3. Place the icosahedron in the scene:

```bash
python3 src/visualize_scene.py
```

This writes `output/icosahedron_scene_full.npz` and displays the object in the
reconstructed scene.

4. Render the object into each frame:

```bash
mkdir -p output/updated_frames
python3 src/render.py
```

This writes rendered frame images to `output/updated_frames/`.

5. Stitch the rendered frames into a video:

```bash
python3 src/make_video.py
```

This writes `output/final_render.mp4`.

## Outputs

Important generated files include:

- `output/inlier_ids.npy`: point indices belonging to the detected plane
- `output/euclidean_transform.npz`: local-to-scene rotation matrix and origin
- `output/icosahedron_scene_full.npz`: object vertices transformed into scene coordinates
- `output/updated_frames/`: rendered image sequence
- `output/final_render.mp4`: final augmented reality result

## Implementation Notes

- `ransac.py` implements plane fitting from scratch by sampling three points,
  computing a plane normal, and retaining the model with the most inliers.
- `transform.py` uses the detected plane inliers to define a local coordinate
  system with the plane centroid as the origin.
- `visualize_scene.py` aligns the icosahedron with the local plane frame and
  transforms it back into COLMAP scene coordinates.
- `render.py` parses COLMAP camera intrinsics and poses, projects the 3D mesh
  into each image, and draws colored triangular faces using a painter's
  algorithm.
- `make_video.py` sorts the rendered frames and writes the final MP4.

## Troubleshooting

- If `transform.py` cannot find `output/inlier_ids.npy`, run `src/ransac.py`
  first.
- If `render.py` cannot find `output/icosahedron_scene_full.npz`, run
  `src/visualize_scene.py` first.
- If no video is produced, confirm that `output/updated_frames/` contains
  rendered `.png`, `.jpg`, or `.jpeg` frames.
