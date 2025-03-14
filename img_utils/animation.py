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
        invert = True if ("_outwards" in self.mode) or ("_down" in self.mode) or ("_right" in self.mode) or ("_counter" in self.mode) else False
        
        if "concentric_circles" in self.mode:
            return self.concentric_circles(frame_number, invert)
        elif "concentric_rectangles" in self.mode:
            return self.concentric_rectangles(frame_number, invert)
        elif "vertical_stripes" in self.mode:
            return self.vertical_stripes(frame_number, invert)
        elif "horizontal_stripes" in self.mode:
            return self.horizontal_stripes(frame_number, invert)
        elif "pushing_segments" in self.mode:
            return self.pushing_segments(frame_number, invert)
        elif "rotating_segments" in self.mode:
            return self.rotating_segments(frame_number, invert)
        else:
            raise ValueError("Unknown mode")

    def concentric_circles(self, frame_number, invert):
        center_x, center_y = self.width / 2, self.height / 2
        y, x = np.ogrid[:self.height, :self.width]
        
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_radius = min(center_x, center_y)
        radius = dist_from_center / max_radius
        
        scale_factor = 1 / self.bands_visible_per_frame
        phase_shift = (frame_number * self.num_shades / self.total_frames) % self.num_shades
        
        if invert:
            phase_shift = -phase_shift

        shade_indices = (radius / scale_factor + phase_shift) % self.num_shades
        shade_indices = np.floor(shade_indices).astype(int)
        shade_indices = np.mod(shade_indices, self.num_shades)  

        frame = self.shades[shade_indices]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def concentric_rectangles(self, frame_number, invert):
        x, y = np.meshgrid(np.linspace(-self.width // 2, self.width // 2, self.width), np.linspace(-self.height // 2, self.height // 2, self.height))

        max_distance = max(self.width, self.height) / 2
        distance_to_center = np.maximum(np.abs(x), np.abs(y))
        scale_factor = max_distance / self.bands_visible_per_frame

        phase_shift = (frame_number * self.num_shades / self.total_frames) % self.num_shades
        
        if invert:
            phase_shift = -phase_shift

        shade_indices = (distance_to_center / scale_factor + phase_shift) % self.num_shades
        shade_indices = np.floor(shade_indices).astype(int)
        shade_indices = np.mod(shade_indices, self.num_shades)

        frame = self.shades[shade_indices]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
    def rotating_segments(self, frame_number, invert):
        x, y = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        angle = np.arctan2(y, x)
        angle = (angle + np.pi) % (2 * np.pi)
        
        phase_shift = -(frame_number * self.num_shades / self.total_frames) % self.num_shades
        
        # Invert the phase shift if needed
        if invert:
            phase_shift = -phase_shift
        
        # Calculate shade indices and apply modulo to ensure valid indices
        shade_indices = ((angle / (2 * np.pi)) * self.num_shades + phase_shift) % self.num_shades
        shade_indices = np.floor(shade_indices).astype(int)
        shade_indices = np.mod(shade_indices, self.num_shades)
        
        frame = self.shades[shade_indices]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


    def vertical_stripes(self, frame_number, invert):
        x = np.linspace(0, self.width - 1, self.width)
        xv = np.tile(x, (self.height, 1))
        
        scale_factor = self.width / self.bands_visible_per_frame
        phase_shift = (frame_number * self.num_shades / self.total_frames) % self.num_shades
        
        if invert:
            phase_shift = -phase_shift

        shade_indices = (xv / scale_factor + phase_shift) % self.num_shades
        shade_indices = np.floor(shade_indices).astype(int)
        shade_indices = np.mod(shade_indices, self.num_shades)
        
        frame = self.shades[shade_indices]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def horizontal_stripes(self, frame_number, invert):
        y = np.linspace(0, self.height - 1, self.height)
        yv = np.tile(y[:, np.newaxis], (1, self.width))
        
        scale_factor = self.height / self.bands_visible_per_frame
        phase_shift = (frame_number * self.num_shades / self.total_frames) % self.num_shades
        
        if invert:
            phase_shift = -phase_shift

        shade_indices = (yv / scale_factor + phase_shift) % self.num_shades
        shade_indices = np.floor(shade_indices).astype(int)
        shade_indices = np.mod(shade_indices, self.num_shades)
        
        frame = self.shades[shade_indices]
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    def pushing_segments(self, frame_number, invert):
        x, y = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        angle = np.arctan2(y, x)
        angle = (angle + np.pi) % (2 * np.pi)  # Normalize angles to [0, 2*pi]
        
        total_rotations = self.total_frames // self.num_shades
        
        # Calculate rotation progress, invert if necessary
        if invert:
            rotation_progress = 1 - (frame_number % total_rotations) / total_rotations
        else:
            rotation_progress = (frame_number % total_rotations) / total_rotations
        
        current_angle = rotation_progress * 2 * np.pi
        
        # Determine the current and next shade indices
        current_shade_idx = (frame_number // total_rotations) % self.num_shades
        next_shade_idx = (current_shade_idx + 1) % self.num_shades
        
        # Create the mask, reverse it if invert is true
        if invert:
            mask = angle > current_angle
        else:
            mask = angle < current_angle
        
        # Create the frame and apply shades based on the mask
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
                "mode": (["concentric_circles_inwards", "concentric_circles_outwards", 
                          "concentric_rectangles_inwards", "concentric_rectangles_outwards", 
                          "rotating_segments_clockwise", "rotating_segments_counter_clockwise", 
                          "pushing_segments_clockwise", "pushing_segments_counter_clockwise", 
                          "vertical_stripes_left", "vertical_stripes_right", 
                          "horizontal_stripes_up", "horizontal_stripes_down"], ),
                "width": ("INT", {"default": 512, "min": 24}),
                "height": ("INT", {"default": 512, "min": 24}),
            }
        }

    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("IMAGE","INT","INT","INT",)
    RETURN_NAMES = ("IMAGE","num_colors","width","height",)
    FUNCTION = "generate_animation"

    def generate_animation(self, total_frames, num_colors, bands_visible_per_frame, angle, mode, width, height):

        animation = Animation(width, height, total_frames, num_colors, bands_visible_per_frame, angle, mode)
        animation_frames = animation.create_animation()

        # Convert the frames to a stack of pytorch tensors:
        animation_frames = np.stack(animation_frames)
        animation_frames = animation_frames.astype(np.float32) / 255.0

        # Convert to torch cpu:
        animation_frames = torch.from_numpy(animation_frames).cpu()

        return animation_frames,num_colors,width,height
    
# Example usage
if __name__ == "__main__":
    width, height = 500, 500  # Canvas size
    total_frames = 48  # Total number of frames in the animation
    num_colors = 3  # Number of discrete colors
    bands_visible_per_frame = 0.9  # Adjust the number of visible bands per frame
    angle = 90  # Rotation angle

    motion_modes = ["concentric_circles_inwards", "concentric_circles_outwards", 
                          "concentric_rectangles_inwards", "concentric_rectangles_outwards", 
                          "rotating_segments_clockwise", "rotating_segments_counter_clockwise", 
                          "pushing_segments_clockwise", "pushing_segments_counter_clockwise", 
                          "vertical_stripes_left", "vertical_stripes_right", 
                          "horizontal_stripes_up", "horizontal_stripes_down"]
    
    motion_modes = ["concentric_circles_inwards", "concentric_circles_outwards"]

    for mode in motion_modes:
        output_file = f'animation_videos/{mode}_{num_colors}_colors_angle_{angle}.mp4'
        animator = Animation(width, height, total_frames, num_colors, bands_visible_per_frame, angle, mode)
        animation_frames = animator.create_animation()

        # save the animation to a file
        animator.save(animation_frames, output_file)
