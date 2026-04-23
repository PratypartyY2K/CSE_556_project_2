import cv2
import os
import glob


def create_video_from_frames(image_folder, output_video_path, fps=10):
    """
    Stitches images from a folder into an MP4 video.
    """
    # Gather and sort the images
    # Supports common extensions like .jpg, .png, .jpeg
    images = [
        img
        for img in os.listdir(image_folder)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Sort alphabetically to ensure chronological sequence
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if not images:
        print(f"Error: No images found in {image_folder}")
        return

    # Get dimensions from the first image
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize VideoWriter
    # 'mp4v' is a widely compatible codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Creating video: {output_video_path}")
    print(f"Stitching {len(images)} frames at {fps} FPS...")

    # Write frames to video
    for image_name in images:
        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)

        # Ensure every frame matches the size of the first one
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))

        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print("Video generation complete.")


def main():
    input_folder = os.path.join("output", "updated_frames")
    output_file = os.path.join("output", "final_render.mp4")

    os.makedirs("assets", exist_ok=True)

    create_video_from_frames(input_folder, output_file, fps=10)


if __name__ == "__main__":
    main()
