import numpy as np
import imageio
from moviepy.editor import ImageSequenceClip
from skimage.color import lab2rgb

def generate_frame(width, height, frame_number, total_frames, num_colors, bands_visible_per_frame, angle):
    # Define a set of equally spaced colors in Lab space
    colors = np.linspace([0, -128, -128], [100, 127, 127], num_colors)
    
    # Create coordinate grid
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    xv, yv = np.meshgrid(x, y)

    # Rotate the coordinate grid
    angle_rad = np.deg2rad(angle)
    xr = xv * np.cos(angle_rad) + yv * np.sin(angle_rad)
    
    # Scale factor to control the width of the bands
    scale_factor = max(width, height) * np.sqrt(2) / bands_visible_per_frame  # Adjusted to cover diagonal distance
    
    # Calculate indices based on rotated coordinates
    phase_shift = (frame_number * num_colors / total_frames) % num_colors
    color_indices = (xr / scale_factor + phase_shift) % num_colors
    color_indices = np.floor(color_indices).astype(int)
    
    # Map indices to colors and convert to RGB
    lab_frame = colors[color_indices]
    frame_rgb = lab2rgb(lab_frame)  # Converts the entire frame in one step
    frame_rgb = (frame_rgb * 255).astype(np.uint8)
    return frame_rgb

def create_animation(width, height, total_frames, num_colors, bands_visible_per_frame, angle, output_file):
    frames = []
    for frame_number in range(total_frames):
        frame = generate_frame(width, height, frame_number, total_frames, num_colors, bands_visible_per_frame, angle)
        frames.append(frame)

    clip = ImageSequenceClip(frames, fps=20)
    clip.write_videofile(output_file, codec='libx264')

if __name__ == "__main__":
    width, height = 500, 500  # Canvas size
    total_frames  = 150  # Total number of frames in the animation
    num_colors    = 4  # Number of discrete colors
    bands_visible_per_frame = 1.25  # Adjust the number of visible bands per frame
    angle = 90 # Rotation angle
    output_file = f'color_transition_{num_colors}_angle_{angle}.mp4'  # Output file name
    create_animation(width, height, total_frames, num_colors, bands_visible_per_frame, angle, output_file)
