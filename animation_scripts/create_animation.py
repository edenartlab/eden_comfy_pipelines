import numpy as np
import imageio
from moviepy.editor import ImageSequenceClip
from skimage.color import lab2rgb

class Animation:
    def __init__(self, width, height, total_frames, num_colors, bands_visible_per_frame, angle, mode):
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.num_colors = num_colors
        self.bands_visible_per_frame = bands_visible_per_frame
        self.angle = angle
        self.mode = mode

        # Define a set of equally spaced colors in Lab space
        self.colors = np.linspace([0, -128, -128], [100, 127, 127], self.num_colors)

    def generate_frame(self, frame_number):
        if self.mode == "panning_rectangles":
            return self.panning_rectangles(frame_number)
        elif self.mode == "concentric_circles":
            return self.concentric_circles(frame_number)
        elif self.mode == "rotating_segments":
            return self.rotating_segments(frame_number)
        elif self.mode == "vertical_stripes":
            return self.vertical_stripes(frame_number)
        elif self.mode == "horizontal_stripes":
            return self.horizontal_stripes(frame_number)
        else:
            raise ValueError("Unknown mode")

    def panning_rectangles(self, frame_number):
        # Create coordinate grid
        x = np.linspace(0, self.width - 1, self.width)
        y = np.linspace(0, self.height - 1, self.height)
        xv, yv = np.meshgrid(x, y)

        # Rotate the coordinate grid
        angle_rad = np.deg2rad(self.angle)
        xr = xv * np.cos(angle_rad) + yv * np.sin(angle_rad)
        
        # Scale factor to control the width of the bands
        scale_factor = max(self.width, self.height) * np.sqrt(2) / self.bands_visible_per_frame
        
        # Calculate indices based on rotated coordinates
        phase_shift = (frame_number * self.num_colors / self.total_frames) % self.num_colors
        color_indices = (xr / scale_factor + phase_shift) % self.num_colors
        color_indices = np.floor(color_indices).astype(int)
        
        # Map indices to colors and convert to RGB
        lab_frame = self.colors[color_indices]
        frame_rgb = lab2rgb(lab_frame)  # Converts the entire frame in one step
        frame_rgb = (frame_rgb * 255).astype(np.uint8)
        return frame_rgb

    def concentric_circles(self, frame_number):
        x, y = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        radius = np.sqrt(x**2 + y**2)
        
        # Scale factor to control the width of the bands
        scale_factor = 1 / self.bands_visible_per_frame
        
        phase_shift = (frame_number * self.num_colors / self.total_frames) % self.num_colors
        color_indices = (radius / scale_factor + phase_shift) % self.num_colors
        color_indices = np.floor(color_indices).astype(int)
        
        lab_frame = self.colors[color_indices]
        frame_rgb = lab2rgb(lab_frame)
        frame_rgb = (frame_rgb * 255).astype(np.uint8)
        return frame_rgb

    def rotating_segments(self, frame_number):
        x, y = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        angle = np.arctan2(y, x)
        
        # Normalize angle to range [0, 2*pi]
        angle = (angle + np.pi) % (2 * np.pi)
        
        phase_shift = (frame_number * self.num_colors / self.total_frames) % self.num_colors
        color_indices = ((angle / (2 * np.pi)) * self.num_colors + phase_shift) % self.num_colors
        color_indices = np.floor(color_indices).astype(int)
        
        lab_frame = self.colors[color_indices]
        frame_rgb = lab2rgb(lab_frame)
        frame_rgb = (frame_rgb * 255).astype(np.uint8)
        return frame_rgb

    def vertical_stripes(self, frame_number):
        x = np.linspace(0, self.width - 1, self.width)
        xv = np.tile(x, (self.height, 1))
        
        scale_factor = self.width / self.bands_visible_per_frame
        phase_shift = (frame_number * self.num_colors / self.total_frames) % self.num_colors
        color_indices = (xv / scale_factor + phase_shift) % self.num_colors
        color_indices = np.floor(color_indices).astype(int)
        
        lab_frame = self.colors[color_indices]
        frame_rgb = lab2rgb(lab_frame)
        frame_rgb = (frame_rgb * 255).astype(np.uint8)
        return frame_rgb

    def horizontal_stripes(self, frame_number):
        y = np.linspace(0, self.height - 1, self.height)
        yv = np.tile(y[:, np.newaxis], (1, self.width))
        
        scale_factor = self.height / self.bands_visible_per_frame
        phase_shift = (frame_number * self.num_colors / self.total_frames) % self.num_colors
        color_indices = (yv / scale_factor + phase_shift) % self.num_colors
        color_indices = np.floor(color_indices).astype(int)
        
        lab_frame = self.colors[color_indices]
        frame_rgb = lab2rgb(lab_frame)
        frame_rgb = (frame_rgb * 255).astype(np.uint8)
        return frame_rgb

    def create_animation(self, output_file):
        frames = []
        for frame_number in range(self.total_frames):
            frame = self.generate_frame(frame_number)
            frames.append(frame)

        clip = ImageSequenceClip(frames, fps=20)
        clip.write_videofile(output_file, codec='libx264')

if __name__ == "__main__":
    width, height = 500, 500  # Canvas size
    total_frames  = 320  # Total number of frames in the animation
    num_colors    = 6  # Number of discrete colors
    bands_visible_per_frame = 1.0  # Adjust the number of visible bands per frame
    angle = 90  # Rotation angle

    # Create animations with different modes
    for mode in ["panning_rectangles", "concentric_circles", "rotating_segments", "vertical_stripes", "horizontal_stripes"]:
        output_file = f'color_transition_{num_colors}_angle_{angle}_{mode}.mp4'
        animation = Animation(width, height, total_frames, num_colors, bands_visible_per_frame, angle, mode)
        animation.create_animation(output_file)
