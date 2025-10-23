#!/usr/bin/env python3
"""
Test script for MaskFromRGB_KMeans node
"""
import torch
from PIL import Image
import os
import subprocess
import numpy as np
import cv2

# Import the node
from img_utils.clustering import MaskFromRGB_KMeans

def load_video_as_tensor(video_path):
    """
    Load a video file (MP4, etc.) and convert it to a tensor in ComfyUI format.
    Returns:
        - tensor of shape (N, H, W, 3) with values in [0, 1]
        - numpy array of original uint8 frames (N, H, W, 3)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frames_uint8 = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV reads in BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_uint8.append(frame_rgb)

        # Convert to tensor format [0, 1]
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frames.append(frame_tensor)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be loaded from {video_path}")

    # Stack frames into a single tensor
    frames_tensor = torch.stack(frames)
    frames_uint8_array = np.stack(frames_uint8)
    return frames_tensor, frames_uint8_array

def load_gif_as_tensor(gif_path):
    """
    Load a GIF file and convert it to a tensor in ComfyUI format.
    Returns:
        - tensor of shape (N, H, W, 3) with values in [0, 1]
        - numpy array of original uint8 frames (N, H, W, 3)
    """
    img = Image.open(gif_path)
    frames = []
    frames_uint8 = []

    try:
        while True:
            # Convert frame to RGB (in case it's in palette mode)
            frame = img.convert('RGB')
            # Convert PIL image to numpy array, then to tensor (proper method)
            frame_np = np.array(frame)
            frames_uint8.append(frame_np)  # Store original uint8
            frame_tensor = torch.from_numpy(frame_np).float() / 255.0
            frames.append(frame_tensor)

            # Move to next frame
            img.seek(img.tell() + 1)
    except EOFError:
        pass  # End of frames

    # Stack frames into a single tensor
    frames_tensor = torch.stack(frames)
    frames_uint8_array = np.stack(frames_uint8)
    return frames_tensor, frames_uint8_array

def load_media_as_tensor(media_path):
    """
    Load a media file (GIF, MP4, etc.) and convert it to a tensor.
    Automatically detects file type and uses appropriate loader.
    Returns:
        - tensor of shape (N, H, W, 3) with values in [0, 1]
        - numpy array of original uint8 frames (N, H, W, 3)
    """
    ext = os.path.splitext(media_path)[1].lower()

    if ext == '.gif':
        return load_gif_as_tensor(media_path)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return load_video_as_tensor(media_path)
    else:
        raise ValueError(f"Unsupported media format: {ext}")

def save_mask_as_image(mask_tensor, output_path):
    """
    Save a mask tensor as an image.
    mask_tensor: (N, H, W) tensor with values in [0, 1]
    """
    # Take first frame if it's a sequence
    if mask_tensor.dim() == 3:
        mask = mask_tensor[0]
    else:
        mask = mask_tensor

    # Convert to PIL image directly from tensor
    mask_scaled = (mask.cpu() * 255).clamp(0, 255).byte()
    img = Image.fromarray(mask_scaled.numpy(), mode='L')
    img.save(output_path)
    print(f"Saved mask to {output_path}")

def save_gif(mask_tensor, output_path, duration=100):
    """
    Save a mask tensor as an animated GIF.
    mask_tensor: (N, H, W) tensor with values in [0, 1]
    """
    frames = []
    for i in range(mask_tensor.shape[0]):
        mask_scaled = (mask_tensor[i].cpu() * 255).clamp(0, 255).byte()
        frames.append(Image.fromarray(mask_scaled.numpy(), mode='L'))

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved animated mask to {output_path}")

def save_rgb_gif(frames_uint8, output_path, duration=100):
    """
    Save RGB frames as an animated GIF.
    frames_uint8: (N, H, W, 3) numpy array with uint8 values
    """
    pil_frames = []
    for i in range(frames_uint8.shape[0]):
        pil_frames.append(Image.fromarray(frames_uint8[i], mode='RGB'))

    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved RGB GIF to {output_path}")

def save_video_grid(input_frames_uint8, masks, combined_mask, output_path, n_clusters, fps=10):
    """
    Create a video grid showing input + all masks side by side and save as MP4.
    input_frames_uint8: (N, H, W, 3) original RGB frames as uint8 numpy array
    masks: list of (N, H, W) mask tensors
    combined_mask: (N, H, W) combined mask tensor
    """
    n_frames = input_frames_uint8.shape[0]
    h, w = input_frames_uint8.shape[1], input_frames_uint8.shape[2]

    # Prepare output directory for temporary frames
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Calculate grid dimensions: input + n_clusters masks + combined
    n_total = 1 + n_clusters + 1
    grid_cols = n_total
    grid_rows = 1

    for frame_idx in range(n_frames):
        # Create grid image
        grid_w = w * grid_cols
        grid_h = h * grid_rows
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        # Add input frame (use original uint8 frames to preserve exact pixel values)
        input_frame = input_frames_uint8[frame_idx]
        grid[0:h, 0:w] = input_frame

        # Add mask frames (convert grayscale to RGB)
        for i in range(n_clusters):
            mask_frame = (masks[i][frame_idx].cpu().numpy() * 255).astype(np.uint8)
            mask_rgb = np.stack([mask_frame] * 3, axis=-1)
            x_offset = (i + 1) * w
            grid[0:h, x_offset:x_offset+w] = mask_rgb

        # Add combined mask
        combined_frame = (combined_mask[frame_idx].cpu().numpy() * 255).astype(np.uint8)
        combined_rgb = np.stack([combined_frame] * 3, axis=-1)
        x_offset = (n_clusters + 1) * w
        grid[0:h, x_offset:x_offset+w] = combined_rgb

        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        Image.fromarray(grid).save(frame_path)

    # Use ffmpeg to create MP4
    print(f"Creating video grid at {output_path}...")
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

    # Clean up temp frames
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

    print(f"Saved video grid to {output_path}")

def main():
    # Parameters from user request
    input_path = "assets/input.mp4"
    n_clusters = 3
    resolution = 128
    feathering_fraction = 0.2
    equalize = 0.0

    print(f"Loading input from {input_path}...")
    image_tensor, image_frames_uint8 = load_media_as_tensor(input_path)
    print(f"Loaded {image_tensor.shape[0]} frames of size {image_tensor.shape[1]}x{image_tensor.shape[2]}")

    # Create node instance
    node = MaskFromRGB_KMeans()

    print(f"\nRunning MaskFromRGB_KMeans with parameters:")
    print(f"  n_color_clusters: {n_clusters}")
    print(f"  clustering_resolution: {resolution}")
    print(f"  feathering_fraction: {feathering_fraction}")
    print(f"  equalize_areas: {equalize}")

    # Execute the node
    results = node.execute(
        image=image_tensor,
        n_color_clusters=n_clusters,
        clustering_resolution=resolution,
        feathering_fraction=feathering_fraction,
        equalize_areas=equalize
    )

    # Unpack results (8 individual masks + combined)
    mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, combined = results

    # Create output directory
    output_dir = "assets/test_output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving outputs to {output_dir}/...")

    # Save loaded input frames as GIF to verify loading is correct
    save_rgb_gif(image_frames_uint8, f"{output_dir}/input.gif")

    # Save individual cluster masks (only the first n_clusters are meaningful)
    masks = [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8]
    for i in range(n_clusters):
        # Save first frame as PNG
        save_mask_as_image(masks[i], f"{output_dir}/mask_{i+1}.png")
        # Save full sequence as GIF
        save_gif(masks[i], f"{output_dir}/mask_{i+1}.gif")

    # Save combined mask
    save_mask_as_image(combined, f"{output_dir}/combined_mask.png")
    save_gif(combined, f"{output_dir}/combined_mask.gif")

    # Create video grid with all outputs
    save_video_grid(
        image_frames_uint8,
        masks[:n_clusters],
        combined,
        f"{output_dir}/output_comparison.mp4",
        n_clusters
    )

    print("\nDone! Check the output directory for results.")
    print(f"Generated {n_clusters} cluster masks plus a combined mask.")
    print(f"Video comparison saved to {output_dir}/output_comparison.mp4")

if __name__ == "__main__":
    main()
