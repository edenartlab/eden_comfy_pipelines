import numpy as np
import os
import cv2
import torch

class Animation:
    def __init__(self, width, height, total_frames, num_shades, bands_visible_per_frame, angle, mode):
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.num_shades = num_shades
        self.bands_visible_per_frame = bands_visible_per_frame
        self.angle = angle
        self.mode = mode

        # Define a set of equally spaced grayscale values from white (255) to black (0)
        self.shades = np.linspace(255, 0, self.num_shades, dtype=np.uint8)

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
        elif self.mode == "progressive_rotating_segment":
            return self.progressive_rotating_segment(frame_number)
        elif self.mode == "concentric_triangles":
            return self.concentric_triangles(frame_number)
        elif self.mode == "concentric_rectangles":
            return self.concentric_rectangles(frame_number)
        else:
            raise ValueError("Unknown mode")
        
    def concentric_triangles(self, frame_number):
        x, y = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        
        frames_per_shade = self.total_frames // self.num_shades
        shade_idx = (frame_number // frames_per_shade) % self.num_shades
        frame_in_shade = frame_number % frames_per_shade
        
        max_radius = np.sqrt(2)
        scale_factor = max_radius / frames_per_shade
        current_radius = frame_in_shade * scale_factor
        
        triangle_shade_idx = shade_idx
        background_shade_idx = (shade_idx + 1) % self.num_shades
        
        height = np.sqrt(3) / 2
        vertices = np.array([
            [0, 2 * height * current_radius],
            [-current_radius, -height * current_radius],
            [current_radius, -height * current_radius]
        ])

        def sign(p1, p2, p3):
            return (p1[:, :, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[:, :, 1] - p3[1])

        pts = np.dstack((x, y))
        d1 = sign(pts, vertices[0], vertices[1])
        d2 = sign(pts, vertices[1], vertices[2])
        d3 = sign(pts, vertices[2], vertices[0])
        
        mask = (d1 >= 0) & (d2 >= 0) & (d3 >= 0)

        frame = np.full((self.height, self.width), self.shades[background_shade_idx], dtype=np.uint8)
        frame[mask] = self.shades[triangle_shade_idx]

        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def concentric_rectangles(self, frame_number):
        x, y = np.meshgrid(np.linspace(-self.width // 2, self.width // 2, self.width), np.linspace(-self.height // 2, self.height // 2, self.height))

        max_distance = max(self.width, self.height) / 2
        distance_to_center = np.maximum(np.abs(x), np.abs(y))
        scale_factor = max_distance / self.bands_visible_per_frame

        phase_shift = (frame_number * self.num_shades / self.total_frames) % self.num_shades
        shade_indices = (distance_to_center / scale_factor + phase_shift) % self.num_shades
        shade_indices = np.floor(shade_indices).astype(int)

        frame = self.shades[shade_indices]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def concentric_circles(self, frame_number):
        x, y = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        radius = np.sqrt(x**2 + y**2)
        
        scale_factor = 1 / self.bands_visible_per_frame
        
        phase_shift = (frame_number * self.num_shades / self.total_frames) % self.num_shades
        shade_indices = (radius / scale_factor + phase_shift) % self.num_shades
        shade_indices = np.floor(shade_indices).astype(int)
        
        frame = self.shades[shade_indices]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def rotating_segments(self, frame_number):
        x, y = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        angle = np.arctan2(y, x)
        
        angle = (angle + np.pi) % (2 * np.pi)
        
        phase_shift = (frame_number * self.num_shades / self.total_frames) % self.num_shades
        shade_indices = ((angle / (2 * np.pi)) * self.num_shades + phase_shift) % self.num_shades
        shade_indices = np.floor(shade_indices).astype(int)
        
        frame = self.shades[shade_indices]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def vertical_stripes(self, frame_number):
        x = np.linspace(0, self.width - 1, self.width)
        xv = np.tile(x, (self.height, 1))
        
        scale_factor = self.width / self.bands_visible_per_frame
        phase_shift = (frame_number * self.num_shades / self.total_frames) % self.num_shades
        shade_indices = (xv / scale_factor + phase_shift) % self.num_shades
        shade_indices = np.floor(shade_indices).astype(int)
        
        frame = self.shades[shade_indices]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def horizontal_stripes(self, frame_number):
        y = np.linspace(0, self.height - 1, self.height)
        yv = np.tile(y[:, np.newaxis], (1, self.width))
        
        scale_factor = self.height / self.bands_visible_per_frame
        phase_shift = (frame_number * self.num_shades / self.total_frames) % self.num_shades
        shade_indices = (yv / scale_factor + phase_shift) % self.num_shades
        shade_indices = np.floor(shade_indices).astype(int)
        
        frame = self.shades[shade_indices]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def progressive_rotating_segment(self, frame_number):
        x, y = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        angle = np.arctan2(y, x)
        
        angle = (angle + np.pi) % (2 * np.pi)
        
        total_rotations = self.total_frames // self.num_shades
        rotation_progress = (frame_number % total_rotations) / total_rotations
        current_angle = rotation_progress * 2 * np.pi
        
        current_shade_idx = (frame_number // total_rotations) % self.num_shades
        next_shade_idx = (current_shade_idx + 1) % self.num_shades
        
        mask = angle < current_angle
        
        frame = np.full((self.height, self.width), self.shades[current_shade_idx], dtype=np.uint8)
        frame[mask] = self.shades[next_shade_idx]
        
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def create_animation(self):
        frames = []
        for frame_number in range(self.total_frames):
            frame = self.generate_frame(frame_number)
            frames.append(frame)

        return frames

    def save(self, frames, output_file, fps=20):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        for frame in frames:
            video.write(frame)

        video.release()


class Animation_RGB_Mask:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 64, "min": 1}),
                "num_colors": ("INT", {"default": 3, "min": 1}),
                "bands_visible_per_frame": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.01}),
                "angle": ("FLOAT", {"default": 0, "min": 0, "max": 360}),
                "mode": (["concentric_circles", "concentric_triangles", "concentric_rectangles", "rotating_segments", "progressive_rotating_segment", "vertical_stripes", "horizontal_stripes"], ),
                "width": ("INT", {"default": 512, "min": 24}),
                "height": ("INT", {"default": 512, "min": 24}),
                "invert_motion": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("IMAGE","INT","INT","INT",)
    RETURN_NAMES = ("IMAGE","num_colors","width","height",)
    FUNCTION = "generate_animation"


    def generate_animation(self, total_frames, num_colors, bands_visible_per_frame, angle, mode, width, height, invert_motion):

        animation = Animation(width, height, total_frames, num_colors, bands_visible_per_frame, angle, mode)
        animation_frames = animation.create_animation()

        # Convert the frames to a stack of pytorch tensors:
        animation_frames = np.stack(animation_frames)
        animation_frames = animation_frames.astype(np.float32) / 255.0

        # Convert to torch cpu:
        animation_frames = torch.from_numpy(animation_frames).cpu()

        if invert_motion:
            animation_frames = animation_frames.flip(0)

        return animation_frames,num_colors,width,height

if __name__ == "__main__":
    width, height = 500, 500  # Canvas size
    total_frames  = 64  # Total number of frames in the animation
    num_colors    = 3  # Number of discrete colors
    bands_visible_per_frame = 1.0  # Adjust the number of visible bands per frame
    angle = 90  # Rotation angle

    # Create animations with different modes
    for mode in ["progressive_rotating_segment"]:
        output_file = f'animation_videos/{mode}_{num_colors}_colors_angle_{angle}.mp4'
        animator = Animation(width, height, total_frames, num_colors, bands_visible_per_frame, angle, mode)
        animation_frames = animator.create_animation()

        # save the animation to a file:
        animator.save(animation_frames, output_file)