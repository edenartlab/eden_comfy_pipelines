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
    

class MaskAnimationGenerator:
    def __init__(self, width, height, total_frames, num_shades):
        self.width = width
        self.height = height
        self.total_frames = total_frames
        if num_shades < 2:
            print(f"Warning: num_shades was {num_shades}, forced to 2 for binary mask generation.")
            self.num_shades = 2
        else:
            self.num_shades = num_shades
        # Shades from dark to light
        self.shades_palette = np.linspace(0, 255, self.num_shades, dtype=np.uint8)

    def _calculate_intensity_from_distance(self, distance_map, gradient_pixel_width, high_intensity, low_intensity):
        if gradient_pixel_width <= 0:
            return np.where(distance_map == 0, high_intensity, low_intensity)
        norm_dist = np.clip(distance_map / gradient_pixel_width, 0.0, 1.0)
        intensity = high_intensity - norm_dist * (high_intensity - low_intensity)
        return intensity

    def _render_frame_with_gradient(self, frame_mask, invert_mask, 
                                    gradient_type, gradient_width_ratio, 
                                    characteristic_dimension_pixels,
                                    motion_params=None):
        
        gradient_pixels = 0.0
        if characteristic_dimension_pixels is not None and characteristic_dimension_pixels > 0:
            gradient_pixels = gradient_width_ratio * characteristic_dimension_pixels
        
        if gradient_type != "none" and gradient_pixels > 0 and self.num_shades == 2:
            print(f"Warning: Gradient type '{gradient_type}' is active with gradient_width_ratio > 0, "
                  f"but 'num_shades' is set to {self.num_shades}. This will produce a binary (black/white) mask. "
                  f"To see a visible gradient, please increase 'num_shades' in the node settings (e.g., to 16, 32, or 64).")

        if gradient_type == "none" or gradient_pixels <= 0:
            active_shade_idx = self.num_shades - 1
            inactive_shade_idx = 0
            if invert_mask:
                active_shade_idx, inactive_shade_idx = inactive_shade_idx, active_shade_idx
            
            output_frame = np.full((self.height, self.width), self.shades_palette[inactive_shade_idx], dtype=np.uint8)
            output_frame[frame_mask] = self.shades_palette[active_shade_idx]
            return output_frame

        float_frame = np.zeros((self.height, self.width), dtype=float)

        if gradient_type == "all_edges":
            if not invert_mask: 
                target_shape_intensity = 1.0
                target_bg_intensity = 0.0
            else: 
                target_shape_intensity = 0.0
                target_bg_intensity = 1.0

            float_frame[frame_mask] = target_shape_intensity 
            float_frame[~frame_mask] = target_bg_intensity

            if not invert_mask:
                dist_map = cv2.distanceTransform((~frame_mask).astype(np.uint8), cv2.DIST_L2, 3)
                relevant_pixels = ~frame_mask
                float_frame[relevant_pixels] = self._calculate_intensity_from_distance(
                    dist_map[relevant_pixels], gradient_pixels, 
                    high_intensity=target_shape_intensity, 
                    low_intensity=target_bg_intensity
                )
            else: 
                dist_map = cv2.distanceTransform(frame_mask.astype(np.uint8), cv2.DIST_L2, 3)
                relevant_pixels = frame_mask 
                float_frame[relevant_pixels] = self._calculate_intensity_from_distance(
                    dist_map[relevant_pixels], gradient_pixels,
                    high_intensity=target_bg_intensity, 
                    low_intensity=target_shape_intensity 
                )
        elif gradient_type == "trailing_edge":
            if motion_params is None or not motion_params.get('axis') or not motion_params.get('direction'):
                print(f"Warning: 'trailing_edge' gradient requires valid motion_params (axis and direction). Falling back to 'all_edges' behavior.")
                # Fallback to all_edges logic
                if not invert_mask: 
                    target_shape_intensity = 1.0
                    target_bg_intensity = 0.0
                else: 
                    target_shape_intensity = 0.0
                    target_bg_intensity = 1.0

                float_frame[frame_mask] = target_shape_intensity 
                float_frame[~frame_mask] = target_bg_intensity

                if not invert_mask:
                    dist_map = cv2.distanceTransform((~frame_mask).astype(np.uint8), cv2.DIST_L2, 3)
                    relevant_pixels = ~frame_mask
                    float_frame[relevant_pixels] = self._calculate_intensity_from_distance(
                        dist_map[relevant_pixels], gradient_pixels, 
                        high_intensity=target_shape_intensity, 
                        low_intensity=target_bg_intensity
                    )
                else: 
                    dist_map = cv2.distanceTransform(frame_mask.astype(np.uint8), cv2.DIST_L2, 3)
                    relevant_pixels = frame_mask 
                    float_frame[relevant_pixels] = self._calculate_intensity_from_distance(
                        dist_map[relevant_pixels], gradient_pixels,
                        high_intensity=target_bg_intensity, 
                        low_intensity=target_shape_intensity
                    )
            else:
                # Determine primary intensities based on invert_mask
                if not invert_mask:
                    shape_peak_intensity = 1.0  # Intensity at the leading edge
                    final_bg_intensity = 0.0    # Intensity of the background and far end of gradient
                else:
                    shape_peak_intensity = 0.0
                    final_bg_intensity = 1.0

                float_frame = np.full((self.height, self.width), final_bg_intensity, dtype=float)

                if motion_params['axis'] == 'radial':
                    y_coords, x_coords = np.ogrid[:self.height, :self.width]
                    center_x, center_y = self.width / 2, self.height / 2
                    dist_from_center_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                    
                    current_radius = characteristic_dimension_pixels 
                    shape_pixels_y, shape_pixels_x = np.where(frame_mask)

                    # Distance from the leading edge (perimeter at current_radius) inwards into the shape
                    dist_from_leading_edge = np.maximum(0, current_radius - dist_from_center_map[shape_pixels_y, shape_pixels_x])
                    
                    norm_dist = np.clip(dist_from_leading_edge / gradient_pixels, 0.0, 1.0) if gradient_pixels > 0 else np.zeros_like(dist_from_leading_edge)
                    intensity_values = shape_peak_intensity - norm_dist * (shape_peak_intensity - final_bg_intensity)
                    float_frame[shape_pixels_y, shape_pixels_x] = intensity_values
                
                elif motion_params['axis'] in ['x', 'y']:
                    for r_idx in range(self.height):
                        for c_idx in range(self.width):
                            if frame_mask[r_idx, c_idx]:
                                depth_from_true_leading_edge = 0
                                max_scan_dist = self.width if motion_params['axis'] == 'x' else self.height
                                
                                for k_scan in range(max_scan_dist):
                                    current_scan_r, current_scan_c = r_idx, c_idx
                                    if motion_params['axis'] == 'x':
                                        current_scan_c = c_idx - motion_params['direction'] * k_scan
                                    else: # axis == 'y'
                                        current_scan_r = r_idx - motion_params['direction'] * k_scan
                                    
                                    if not (0 <= current_scan_r < self.height and 0 <= current_scan_c < self.width) or \
                                       not frame_mask[current_scan_r, current_scan_c]:
                                        depth_from_true_leading_edge = k_scan # k_scan is number of steps to hit boundary
                                        break
                                else: # Should not happen if pixel is in mask unless mask is full line
                                    depth_from_true_leading_edge = max_scan_dist 
                                
                                # Convert depth (distance to boundary) to effective depth from leading surface (0 for leading pixel)
                                effective_depth = max(0, depth_from_true_leading_edge)

                                norm_dist = np.clip(effective_depth / gradient_pixels, 0.0, 1.0) if gradient_pixels > 0 else 0.0
                                current_intensity = shape_peak_intensity - norm_dist * (shape_peak_intensity - final_bg_intensity)
                                float_frame[r_idx, c_idx] = current_intensity
                else:
                    print(f"Warning: Unknown motion axis '{motion_params['axis']}' for trailing_edge. Applying solid color to shape.")
                    float_frame[frame_mask] = shape_peak_intensity
        else:
            print(f"Warning: Unknown gradient type '{gradient_type}'. No gradient applied.")
            active_shade_idx = self.num_shades - 1; inactive_shade_idx = 0
            if invert_mask: active_shade_idx, inactive_shade_idx = inactive_shade_idx, active_shade_idx
            output_frame = np.full((self.height, self.width), self.shades_palette[inactive_shade_idx], dtype=np.uint8)
            output_frame[frame_mask] = self.shades_palette[active_shade_idx]
            return output_frame

        indices = np.round(float_frame * (self.num_shades - 1)).astype(int)
        indices = np.clip(indices, 0, self.num_shades - 1)
        final_colored_frame = self.shades_palette[indices]
        return final_colored_frame

    def moving_band(self, frame_number, orientation_is_vertical, direction_is_forward, band_thickness_ratio, invert_mask,
                    gradient_type, gradient_width_ratio):
        progress = frame_number / self.total_frames
        frame_mask = np.zeros((self.height, self.width), dtype=bool)
        characteristic_dimension_pixels = 0
        motion_params = {}

        if not orientation_is_vertical: # Horizontal band, moving vertically
            band_h_pixels = int(band_thickness_ratio * self.height)
            characteristic_dimension_pixels = band_h_pixels
            pos = progress * (self.height + band_h_pixels)
            motion_params = {'axis': 'y', 'direction': +1 if direction_is_forward else -1}

            if direction_is_forward: # Top to Bottom
                y1 = pos - band_h_pixels
                y2 = pos         
            else: # Bottom to Top
                y1 = self.height - pos 
                y2 = self.height - (pos - band_h_pixels)
            
            y1_c = int(np.clip(y1, 0, self.height))
            y2_c = int(np.clip(y2, 0, self.height))
            if y2_c > y1_c:
                 frame_mask[y1_c:y2_c, :] = True
        else: # Vertical band, moving horizontally
            band_w_pixels = int(band_thickness_ratio * self.width)
            characteristic_dimension_pixels = band_w_pixels
            pos = progress * (self.width + band_w_pixels)
            motion_params = {'axis': 'x', 'direction': +1 if direction_is_forward else -1}

            if direction_is_forward: # Left to Right
                x1 = pos - band_w_pixels
                x2 = pos         
            else: # Right to Left
                x1 = self.width - pos
                x2 = self.width - (pos - band_w_pixels)

            x1_c = int(np.clip(x1, 0, self.width))
            x2_c = int(np.clip(x2, 0, self.width))
            if x2_c > x1_c:
                frame_mask[:, x1_c:x2_c] = True
        
        return self._render_frame_with_gradient(frame_mask, invert_mask, gradient_type, gradient_width_ratio, characteristic_dimension_pixels, motion_params)

    def sine_wave(self, frame_number, wave_axis_is_vertical, motion_is_forward, amplitude_ratio, frequency, wave_thickness_ratio, invert_mask,
                  gradient_type, gradient_width_ratio):
        progress = frame_number / self.total_frames
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        frame_mask = np.zeros((self.height, self.width), dtype=bool)
        characteristic_dimension_pixels = 0
        motion_params = {}

        phase_movement = progress * 2 * np.pi
        if not motion_is_forward:
            phase_movement = -phase_movement

        if wave_axis_is_vertical: 
            amplitude_pixels = amplitude_ratio * self.width / 2 
            wave_thickness_pixels = wave_thickness_ratio * self.width
            characteristic_dimension_pixels = wave_thickness_pixels
            motion_params = {'axis': 'x', 'direction': +1 if motion_is_forward else -1}
            
            wave_center_x = (self.width / 2 + 
                             amplitude_pixels * np.sin(2 * np.pi * frequency * y_coords / self.height + phase_movement))
            frame_mask = np.abs(x_coords - wave_center_x) < (wave_thickness_pixels / 2)
        else: 
            amplitude_pixels = amplitude_ratio * self.height / 2
            wave_thickness_pixels = wave_thickness_ratio * self.height
            characteristic_dimension_pixels = wave_thickness_pixels
            motion_params = {'axis': 'y', 'direction': +1 if motion_is_forward else -1}

            wave_center_y = (self.height / 2 +
                             amplitude_pixels * np.sin(2 * np.pi * frequency * x_coords / self.width + phase_movement))
            frame_mask = np.abs(y_coords - wave_center_y) < (wave_thickness_pixels / 2)
        
        return self._render_frame_with_gradient(frame_mask, invert_mask, gradient_type, gradient_width_ratio, characteristic_dimension_pixels, motion_params)

    def expanding_contracting_circle(self, frame_number, is_expanding, invert_mask,
                                     gradient_type, gradient_width_ratio):
        progress = frame_number / self.total_frames
        center_x, center_y = self.width / 2, self.height / 2
        max_radius = np.sqrt((self.width/2)**2 + (self.height/2)**2) 
        motion_params = {'axis': 'radial', 'direction': +1 if is_expanding else -1}
        
        current_radius = 0.0
        if is_expanding:
            current_radius = progress * max_radius
        else: 
            current_radius = (1.0 - progress) * max_radius
            
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        
        frame_mask = dist_from_center <= current_radius
        characteristic_dimension_pixels = current_radius
        return self._render_frame_with_gradient(frame_mask, invert_mask, gradient_type, gradient_width_ratio, characteristic_dimension_pixels, motion_params)


class AnimatedShapeMaskNode:
    CATEGORY = "Eden 🌱/Masks"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "num_shades", "width", "height",)
    FUNCTION = "generate_mask_animation"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 24, "step": 8}),
                "height": ("INT", {"default": 512, "min": 24, "step": 8}),
                "total_frames": ("INT", {"default": 64, "min": 1}),
                "num_shades": ("INT", {"default": 2, "min": 2, "max": 256}), # Min 2 for black/white
                "mode": (["moving_band_horizontal_td", "moving_band_horizontal_bu",
                          "moving_band_vertical_lr", "moving_band_vertical_rl",
                          "sine_wave_vertical_lr", "sine_wave_vertical_rl",
                          "sine_wave_horizontal_td", "sine_wave_horizontal_bu",
                          "expanding_circle", "contracting_circle"], ),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "band_thickness_ratio": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
                "wave_amplitude_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "wave_frequency": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "wave_thickness_ratio": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01}),
                "gradient_type": (["none", "all_edges", "trailing_edge"], {"default": "none"}),
                "gradient_width_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def generate_mask_animation(self, width, height, total_frames, num_shades, mode, invert_mask, 
                               band_thickness_ratio, wave_amplitude_ratio, wave_frequency, wave_thickness_ratio,
                               gradient_type, gradient_width_ratio):
        
        generator = MaskAnimationGenerator(width, height, total_frames, num_shades)
        animation_frames_list = []

        for i in range(total_frames):
            frame = None
            if mode.startswith("moving_band_horizontal"): # Horizontal band, moves vertically
                orientation_is_vertical = False
                direction_is_forward = "_td" in mode # True for Top-Down
                frame = generator.moving_band(i, orientation_is_vertical, direction_is_forward, band_thickness_ratio, invert_mask,
                                              gradient_type, gradient_width_ratio)
            elif mode.startswith("moving_band_vertical"): # Vertical band, moves horizontally
                orientation_is_vertical = True
                direction_is_forward = "_lr" in mode # True for Left-Right
                frame = generator.moving_band(i, orientation_is_vertical, direction_is_forward, band_thickness_ratio, invert_mask,
                                              gradient_type, gradient_width_ratio)
            elif mode.startswith("sine_wave_vertical"): # Wave oscillates along X (vertical appearance), moves horizontally
                wave_axis_is_vertical = True
                motion_is_forward = "_lr" in mode # True for Left-Right
                frame = generator.sine_wave(i, wave_axis_is_vertical, motion_is_forward, wave_amplitude_ratio, wave_frequency, wave_thickness_ratio, invert_mask,
                                            gradient_type, gradient_width_ratio)
            elif mode.startswith("sine_wave_horizontal"): # Wave oscillates along Y (horizontal appearance), moves vertically
                wave_axis_is_vertical = False
                motion_is_forward = "_td" in mode # True for Top-Down
                frame = generator.sine_wave(i, wave_axis_is_vertical, motion_is_forward, wave_amplitude_ratio, wave_frequency, wave_thickness_ratio, invert_mask,
                                            gradient_type, gradient_width_ratio)
            elif mode == "expanding_circle":
                frame = generator.expanding_contracting_circle(i, is_expanding=True, invert_mask=invert_mask,
                                                               gradient_type=gradient_type, gradient_width_ratio=gradient_width_ratio)
            elif mode == "contracting_circle":
                frame = generator.expanding_contracting_circle(i, is_expanding=False, invert_mask=invert_mask,
                                                               gradient_type=gradient_type, gradient_width_ratio=gradient_width_ratio)
            
            if frame is not None:
                animation_frames_list.append(frame)
            else:
                # This case should ideally not be reached if all modes are handled.
                print(f"Warning: Mode '{mode}' was not handled. Appending a blank frame.")
                error_frame = np.full((height, width), generator.shades_palette[0], dtype=np.uint8)
                animation_frames_list.append(error_frame)


        if not animation_frames_list and total_frames > 0 :
            print("Warning: No frames were generated. Returning blank animation.")
            fallback_frame = np.full((height, width), generator.shades_palette[0], dtype=np.uint8)
            if invert_mask and num_shades > 0 : fallback_frame.fill(generator.shades_palette[-1])
            animation_frames_list = [fallback_frame for _ in range(total_frames)]
        elif not animation_frames_list and total_frames == 0: # Handle total_frames = 0 case
             animation_frames_list = [np.full((height, width), generator.shades_palette[0], dtype=np.uint8)]


        animation_frames_np = np.stack(animation_frames_list)
        animation_frames_tensor = torch.from_numpy(animation_frames_np).float() / 255.0
        animation_frames_tensor = animation_frames_tensor.unsqueeze(-1) # Shape: (N, H, W, 1)
        animation_frames_tensor = animation_frames_tensor.repeat(1, 1, 1, 3) # Shape: (N, H, W, 3)
        
        return (animation_frames_tensor, num_shades, width, height)

if __name__ == "__main__":
    # Helper function to save animation.
    # Handles grayscale (N, H, W, 1 or N, H, W) or 3-channel (N, H, W, 3) tensors.
    # If single channel, converts to BGR. If 3-channel, assumes it's BGR-compatible (e.g., R=G=B).
    def save_grayscale_animation(frames_tensor, output_file, fps=20):
        # Ensure tensor is on CPU and in numpy format, scaled to 0-255 uint8
        frames_np = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        output_dir = os.path.dirname(output_file)
        if output_dir: # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
        
        if not frames_np.size: # Check if frames_np is empty (e.g. total_frames = 0)
            print(f"No frames to save for {output_file}.")
            return

        # Determine frame dimensions and if conversion is needed
        height, width = 0, 0
        needs_conversion_to_bgr = False

        if frames_np.ndim == 4 and frames_np.shape[-1] == 1: # Input is (N, H, W, 1)
            frames_np = frames_np.squeeze(-1) # Now (N, H, W)
            # Check if squeeze resulted in an empty or non-2D array for frames_np[0]
            if frames_np.ndim != 3 or frames_np.shape[0] == 0: # Should be (N,H,W)
                print(f"Unsupported frame format after squeeze: {frames_np.shape}. Cannot save video for {output_file}.")
                return
            height, width = frames_np[0].shape
            needs_conversion_to_bgr = True
        elif frames_np.ndim == 3: # Input is (N, H, W)
            if frames_np.shape[0] == 0:
                print(f"No frames to save (empty first dimension): {output_file}.")
                return
            height, width = frames_np[0].shape
            needs_conversion_to_bgr = True
        elif frames_np.ndim == 4 and frames_np.shape[-1] == 3: # Input is (N, H, W, 3)
            if frames_np.shape[0] == 0:
                print(f"No frames to save (empty first dimension): {output_file}.")
                return
            height, width, channels = frames_np[0].shape
            if channels != 3:
                 print(f"Unsupported frame format: {channels} channels. Expected 3. Cannot save video for {output_file}.")
                 return
            needs_conversion_to_bgr = False # Assume it's already BGR or R=G=B
        else:
            print(f"Unsupported frame format: {frames_np.shape}. Cannot save video for {output_file}.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        for frame_data in frames_np:
            if needs_conversion_to_bgr:
                # frame_data is (H, W) grayscale
                frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
                video.write(frame_bgr)
            else:
                # frame_data is (H, W, 3), assumed BGR compatible
                video.write(frame_data)
        video.release()
        print(f"Saved animation to {output_file}")

    # Test parameters
    test_width = 256
    test_height = 128
    test_total_frames = 60
    # Default num_shades for tests not specifically overriding it for gradient visibility
    default_test_num_shades = 2 
    test_num_shades_for_gradient = 64 # Use more shades to see gradients
    
    node_instance = AnimatedShapeMaskNode()

    # Clean up previous test structure if it's too verbose or less relevant now
    # Focus on new gradient tests for moving_band
    output_folder = "animation_videos/shape_masks_gradient_tests"
    os.makedirs(output_folder, exist_ok=True)

    gradient_test_params_moving_band = {
        "moving_band_vl_lr_no_gradient": {
            "mode": "moving_band_vertical_lr", "invert_mask": False, 
            "band_thickness_ratio": 0.25, 
            "gradient_type": "none", "gradient_width_ratio": 0.0, 
            "num_shades_override": default_test_num_shades
        },
        "moving_band_vl_lr_all_edges_0.25": {
            "mode": "moving_band_vertical_lr", "invert_mask": False, 
            "band_thickness_ratio": 0.25, 
            "gradient_type": "all_edges", "gradient_width_ratio": 0.25, 
            "num_shades_override": test_num_shades_for_gradient
        },
        "moving_band_vl_lr_all_edges_0.50": {
            "mode": "moving_band_vertical_lr", "invert_mask": False, 
            "band_thickness_ratio": 0.25, 
            "gradient_type": "all_edges", "gradient_width_ratio": 0.50, 
            "num_shades_override": test_num_shades_for_gradient
        },
        "moving_band_vl_lr_all_edges_0.25_inverted": {
            "mode": "moving_band_vertical_lr", "invert_mask": True, 
            "band_thickness_ratio": 0.25, 
            "gradient_type": "all_edges", "gradient_width_ratio": 0.25, 
            "num_shades_override": test_num_shades_for_gradient
        },
        "moving_band_vl_lr_trailing_edge_0.25_fallback": {
            "mode": "moving_band_vertical_lr", "invert_mask": False, 
            "band_thickness_ratio": 0.25, 
            "gradient_type": "trailing_edge", "gradient_width_ratio": 0.25, 
            "num_shades_override": test_num_shades_for_gradient
        },
        # Example of a different shape with gradient
        "expanding_circle_all_edges_0.3_64shades": {
            "mode": "expanding_circle", "invert_mask": False, 
            "band_thickness_ratio": 0.2, # Not used by circle, but provide for completeness
            "gradient_type": "all_edges", "gradient_width_ratio": 0.3,
            "num_shades_override": test_num_shades_for_gradient
        }
    }

    # Common parameters not used by all modes, but required by the function signature
    default_wave_params = {
        "wave_amplitude_ratio": 0.1, 
        "wave_frequency": 2.0, 
        "wave_thickness_ratio": 0.05
    }

    for name, params in gradient_test_params_moving_band.items():
        current_num_shades = params.get("num_shades_override", default_test_num_shades)
        print(f"Generating: {name} with {current_num_shades} shades, gradient: {params['gradient_type']}, width: {params['gradient_width_ratio']}")
        
        image_tensor, _, _, _ = node_instance.generate_mask_animation(
            width=test_width,
            height=test_height,
            total_frames=test_total_frames,
            num_shades=current_num_shades,
            mode=params["mode"],
            invert_mask=params["invert_mask"],
            band_thickness_ratio=params["band_thickness_ratio"],
            # Fill in other params with defaults if not specified, or use fixed defaults
            wave_amplitude_ratio=params.get("wave_amplitude_ratio", default_wave_params["wave_amplitude_ratio"]),
            wave_frequency=params.get("wave_frequency", default_wave_params["wave_frequency"]),
            wave_thickness_ratio=params.get("wave_thickness_ratio", default_wave_params["wave_thickness_ratio"]),
            gradient_type=params["gradient_type"],
            gradient_width_ratio=params["gradient_width_ratio"]
        )
        
        output_file_path = os.path.join(output_folder, f"{name}.mp4")
        save_grayscale_animation(image_tensor, output_file_path, fps=15)

    print(f"All test animations saved in ./{output_folder}")
