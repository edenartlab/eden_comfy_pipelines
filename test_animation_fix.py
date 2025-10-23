#!/usr/bin/env python3
"""
Test script to verify the Animation_RGB_Mask fix for smooth color transitions
"""
import sys
import os
import numpy as np
from PIL import Image

# Import the node
from img_utils.animation import Animation_RGB_Mask

def save_animation_as_gif(frames_tensor, output_path, duration=80):
    """
    Save animation tensor as a GIF
    frames_tensor: (N, H, W, 3) tensor with values in [0, 1]
    """
    frames_np = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)

    pil_frames = []
    for i in range(frames_np.shape[0]):
        pil_frames.append(Image.fromarray(frames_np[i], mode='RGB'))

    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Saved animation to {output_path}")

def main():
    # Create output directory
    output_dir = "assets/test_output"
    os.makedirs(output_dir, exist_ok=True)

    # Test parameters - same as likely used for input.gif
    node = Animation_RGB_Mask()

    print("Testing concentric_circles_inwards with 3 colors...")
    frames, num_colors, width, height = node.generate_animation(
        total_frames=192,
        num_colors=3,
        bands_visible_per_frame=1.0,
        angle=0,
        mode="concentric_circles_inwards",
        width=520,
        height=592
    )

    print(f"Generated {frames.shape[0]} frames of size {width}x{height}")
    print(f"Frame tensor shape: {frames.shape}")

    # Save as GIF
    output_path = os.path.join(output_dir, "test_smooth_animation.gif")
    save_animation_as_gif(frames, output_path, duration=80)

    # Also convert to MP4 for comparison
    mp4_path = os.path.join(output_dir, "test_smooth_animation.mp4")
    print(f"\nConverting to MP4: {mp4_path}")
    os.system(f'ffmpeg -y -i "{output_path}" -c:v libx264 -pix_fmt yuv420p -vf "fps=12" "{mp4_path}" 2>&1 | tail -5')

    print("\n✓ Done! Check the output files:")
    print(f"  - {output_path}")
    print(f"  - {mp4_path}")
    print("\nThe animation should now have smooth color transitions without sudden flips!")

if __name__ == "__main__":
    main()
