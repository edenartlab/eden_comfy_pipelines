import logging
import sys

import numpy as np
import torch
import comfy.utils


def _gray_to_image(frames):
    """(N, H, W) float array in 0..1 -> ComfyUI IMAGE (N, H, W, 3)."""
    return torch.from_numpy(frames).unsqueeze(-1).repeat(1, 1, 1, 3)


class Animation:
    def __init__(self, width, height, total_frames, num_shades, bands_visible_per_frame, mode):
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.num_shades = num_shades
        self.bands_visible_per_frame = bands_visible_per_frame
        self.mode = mode
        # Equally spaced grayscale values from white (255) to black (0)
        self.shades = np.linspace(255, 0, self.num_shades, dtype=np.uint8)
        self.invert = any(k in mode for k in ("_outwards", "_down", "_right", "_counter"))

    def _base_map(self):
        """Per-pixel band coordinate; the frame's shade index is floor(base + phase_shift) mod num_shades."""
        mode, w, h = self.mode, self.width, self.height
        if "concentric_circles" in mode:
            center_x, center_y = w / 2, h / 2
            y, x = np.ogrid[:h, :w]
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2) / min(center_x, center_y)
            return radius / (1 / self.bands_visible_per_frame)
        if "concentric_rectangles" in mode:
            x, y = np.meshgrid(np.linspace(-w // 2, w // 2, w), np.linspace(-h // 2, h // 2, h))
            scale_factor = (max(w, h) / 2) / self.bands_visible_per_frame
            return np.maximum(np.abs(x), np.abs(y)) / scale_factor
        if "vertical_stripes" in mode:
            xv = np.tile(np.linspace(0, w - 1, w), (h, 1))
            return xv / (w / self.bands_visible_per_frame)
        if "horizontal_stripes" in mode:
            yv = np.tile(np.linspace(0, h - 1, h)[:, np.newaxis], (1, w))
            return yv / (h / self.bands_visible_per_frame)
        if "rotating_segments" in mode:
            return (self._angle_map() / (2 * np.pi)) * self.num_shades
        raise ValueError(f"Unknown mode {mode}")

    def _angle_map(self):
        x, y = np.meshgrid(np.linspace(-1, 1, self.width), np.linspace(-1, 1, self.height))
        return (np.arctan2(y, x) + np.pi) % (2 * np.pi)

    def _phase_shift(self, frame_number):
        shift = frame_number * self.num_shades / self.total_frames
        phase_shift = (-shift if "rotating_segments" in self.mode else shift) % self.num_shades
        return -phase_shift if self.invert else phase_shift

    def pushing_segments(self, angle, frame_number):
        # max(1, ...) avoids a ZeroDivisionError when total_frames < num_shades
        total_rotations = max(1, self.total_frames // self.num_shades)
        rotation_progress = (frame_number % total_rotations) / total_rotations
        if self.invert:
            rotation_progress = 1 - rotation_progress
        current_angle = rotation_progress * 2 * np.pi

        current_shade_idx = (frame_number // total_rotations) % self.num_shades
        next_shade_idx = (current_shade_idx + 1) % self.num_shades
        mask = angle > current_angle if self.invert else angle < current_angle
        return np.where(mask, next_shade_idx, current_shade_idx)

    def create_animation(self):
        """Returns (total_frames, H, W) float32 frames in 0..1."""
        shades = self.shades.astype(np.float32) / 255.0
        frames = np.empty((self.total_frames, self.height, self.width), dtype=np.float32)
        pbar = comfy.utils.ProgressBar(self.total_frames)
        if "pushing_segments" in self.mode:
            angle = self._angle_map()
            for f in range(self.total_frames):
                frames[f] = shades[self.pushing_segments(angle, f)]
                pbar.update(1)
            return frames

        base = self._base_map()
        for f in range(self.total_frames):
            shade_indices = np.floor((base + self._phase_shift(f)) % self.num_shades).astype(int)
            frames[f] = shades[np.mod(shade_indices, self.num_shades)]
            pbar.update(1)
        return frames


class Animation_RGB_Mask:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 64, "min": 1, "max": sys.maxsize, "tooltip": "Number of frames; the pattern completes one full cycle over this length, so the clip loops."}),
                "num_colors": ("INT", {"default": 3, "min": 1, "max": sys.maxsize, "tooltip": "Number of gray levels (bands) in the pattern."}),
                "bands_visible_per_frame": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.01, "tooltip": "How many bands fit across the frame (ignored by the segment modes)."}),
                "angle": ("FLOAT", {"default": 0, "min": 0, "max": 360, "tooltip": "Unused (kept for workflow compatibility)."}),
                "mode": (["concentric_circles_inwards", "concentric_circles_outwards",
                          "concentric_rectangles_inwards", "concentric_rectangles_outwards",
                          "rotating_segments_clockwise", "rotating_segments_counter_clockwise",
                          "pushing_segments_clockwise", "pushing_segments_counter_clockwise",
                          "vertical_stripes_left", "vertical_stripes_right",
                          "horizontal_stripes_up", "horizontal_stripes_down"], {"tooltip": "Shape and direction of the moving bands."}),
                "width": ("INT", {"default": 512, "min": 24, "max": sys.maxsize, "tooltip": "Output width in pixels."}),
                "height": ("INT", {"default": 512, "min": 24, "max": sys.maxsize, "tooltip": "Output height in pixels."}),
            }
        }

    CATEGORY = "Eden 🌱/Mask"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "num_colors", "width", "height",)
    FUNCTION = "generate_animation"
    DESCRIPTION = "Generates a looping grayscale band animation (circles, rectangles, stripes, segments), typically used as a multi-region mask video."
    OUTPUT_TOOLTIPS = ("Grayscale animation frames.", "Number of gray levels.", "Width in pixels.", "Height in pixels.")

    def generate_animation(self, total_frames, num_colors, bands_visible_per_frame, angle, mode, width, height):
        frames = Animation(width, height, total_frames, num_colors, bands_visible_per_frame, mode).create_animation()
        return _gray_to_image(frames), num_colors, width, height


def _run_length(mask, axis, direction):
    """For each pixel, the number of consecutive mask pixels ending at it when walking
    against `direction` along `axis` (1 for the pixel on the leading edge)."""
    if direction < 0:
        mask = np.flip(mask, axis)
    idx = np.arange(mask.shape[axis]).reshape((-1, 1) if axis == 0 else (1, -1))
    last_gap = np.maximum.accumulate(np.where(mask, -1, idx), axis=axis)
    run = idx - last_gap
    return np.flip(run, axis) if direction < 0 else run


class MaskAnimationGenerator:
    def __init__(self, width, height, total_frames, num_shades):
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.num_shades = max(2, num_shades)
        # Shades from dark to light
        self.shades_palette = np.linspace(0, 255, self.num_shades, dtype=np.uint8)

    @staticmethod
    def _intensity_from_distance(distance_map, gradient_pixels, high_intensity, low_intensity):
        norm_dist = np.clip(distance_map / gradient_pixels, 0.0, 1.0)
        return high_intensity - norm_dist * (high_intensity - low_intensity)

    def _render_frame_with_gradient(self, frame_mask, invert_mask, gradient_type, gradient_width_ratio,
                                    characteristic_dimension_pixels, motion_params):
        import cv2

        gradient_pixels = 0.0
        if characteristic_dimension_pixels > 0:
            gradient_pixels = gradient_width_ratio * characteristic_dimension_pixels

        if gradient_type == "none" or gradient_pixels <= 0:
            active, inactive = self.shades_palette[-1], self.shades_palette[0]
            if invert_mask:
                active, inactive = inactive, active
            return np.where(frame_mask, active, inactive).astype(np.uint8)

        shape_intensity, bg_intensity = (0.0, 1.0) if invert_mask else (1.0, 0.0)

        if gradient_type == "all_edges":
            # The gradient falls off outside the shape (or inside it when inverted)
            relevant_pixels = frame_mask if invert_mask else ~frame_mask
            float_frame = np.where(frame_mask, shape_intensity, bg_intensity)
            dist_map = cv2.distanceTransform(relevant_pixels.astype(np.uint8), cv2.DIST_L2, 3)
            high, low = (bg_intensity, shape_intensity) if invert_mask else (shape_intensity, bg_intensity)
            float_frame[relevant_pixels] = self._intensity_from_distance(dist_map[relevant_pixels], gradient_pixels, high, low)
        else:  # trailing_edge
            float_frame = np.full((self.height, self.width), bg_intensity, dtype=float)
            if motion_params['axis'] == 'radial':
                y_coords, x_coords = np.ogrid[:self.height, :self.width]
                dist_from_center = np.sqrt((x_coords - self.width / 2)**2 + (y_coords - self.height / 2)**2)
                depth = np.maximum(0, characteristic_dimension_pixels - dist_from_center[frame_mask])
            else:
                axis = 1 if motion_params['axis'] == 'x' else 0
                depth = _run_length(frame_mask, axis, motion_params['direction'])[frame_mask]
            float_frame[frame_mask] = self._intensity_from_distance(depth, gradient_pixels, shape_intensity, bg_intensity)

        indices = np.clip(np.round(float_frame * (self.num_shades - 1)).astype(int), 0, self.num_shades - 1)
        return self.shades_palette[indices]

    def moving_band(self, frame_number, orientation_is_vertical, direction_is_forward, band_thickness_ratio, invert_mask,
                    gradient_type, gradient_width_ratio):
        progress = frame_number / self.total_frames
        frame_mask = np.zeros((self.height, self.width), dtype=bool)
        size = self.width if orientation_is_vertical else self.height
        band_pixels = int(band_thickness_ratio * size)
        pos = progress * (size + band_pixels)
        if direction_is_forward:
            lo, hi = pos - band_pixels, pos
        else:
            lo, hi = size - pos, size - (pos - band_pixels)
        lo, hi = int(np.clip(lo, 0, size)), int(np.clip(hi, 0, size))
        if orientation_is_vertical:  # vertical band moving horizontally
            frame_mask[:, lo:hi] = True
        else:
            frame_mask[lo:hi, :] = True

        motion_params = {'axis': 'x' if orientation_is_vertical else 'y', 'direction': 1 if direction_is_forward else -1}
        return self._render_frame_with_gradient(frame_mask, invert_mask, gradient_type, gradient_width_ratio, band_pixels, motion_params)

    def sine_wave(self, frame_number, wave_axis_is_vertical, motion_is_forward, amplitude_ratio, frequency, wave_thickness_ratio, invert_mask,
                  gradient_type, gradient_width_ratio):
        progress = frame_number / self.total_frames
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        phase_movement = progress * 2 * np.pi
        if not motion_is_forward:
            phase_movement = -phase_movement

        if wave_axis_is_vertical:
            amplitude_pixels = amplitude_ratio * self.width / 2
            wave_thickness_pixels = wave_thickness_ratio * self.width
            wave_center_x = self.width / 2 + amplitude_pixels * np.sin(2 * np.pi * frequency * y_coords / self.height + phase_movement)
            frame_mask = np.abs(x_coords - wave_center_x) < (wave_thickness_pixels / 2)
        else:
            amplitude_pixels = amplitude_ratio * self.height / 2
            wave_thickness_pixels = wave_thickness_ratio * self.height
            wave_center_y = self.height / 2 + amplitude_pixels * np.sin(2 * np.pi * frequency * x_coords / self.width + phase_movement)
            frame_mask = np.abs(y_coords - wave_center_y) < (wave_thickness_pixels / 2)

        motion_params = {'axis': 'x' if wave_axis_is_vertical else 'y', 'direction': 1 if motion_is_forward else -1}
        return self._render_frame_with_gradient(frame_mask, invert_mask, gradient_type, gradient_width_ratio, wave_thickness_pixels, motion_params)

    def expanding_contracting_circle(self, frame_number, is_expanding, invert_mask, gradient_type, gradient_width_ratio):
        progress = frame_number / self.total_frames
        max_radius = np.sqrt((self.width/2)**2 + (self.height/2)**2)
        current_radius = progress * max_radius if is_expanding else (1.0 - progress) * max_radius

        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        frame_mask = np.sqrt((x_grid - self.width / 2)**2 + (y_grid - self.height / 2)**2) <= current_radius
        motion_params = {'axis': 'radial', 'direction': 1 if is_expanding else -1}
        return self._render_frame_with_gradient(frame_mask, invert_mask, gradient_type, gradient_width_ratio, current_radius, motion_params)


class AnimatedShapeMaskNode:
    CATEGORY = "Eden 🌱/Mask"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "num_shades", "width", "height",)
    FUNCTION = "generate_mask_animation"
    DESCRIPTION = "Generates an animated shape mask (moving band, sine wave or growing/shrinking circle), optionally with soft gradient edges."
    OUTPUT_TOOLTIPS = ("Mask animation frames (grayscale IMAGE).", "Number of gray levels used.", "Width in pixels.", "Height in pixels.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 24, "step": 8, "max": sys.maxsize, "tooltip": "Output width in pixels."}),
                "height": ("INT", {"default": 512, "min": 24, "step": 8, "max": sys.maxsize, "tooltip": "Output height in pixels."}),
                "total_frames": ("INT", {"default": 64, "min": 1, "max": sys.maxsize, "tooltip": "Number of frames in the animation."}),
                "num_shades": ("INT", {"default": 2, "min": 2, "max": 256, "tooltip": "Gray levels. 2 = hard black/white mask; use 16+ to see gradients."}),
                "mode": (["moving_band_horizontal_td", "moving_band_horizontal_bu",
                          "moving_band_vertical_lr", "moving_band_vertical_rl",
                          "sine_wave_vertical_lr", "sine_wave_vertical_rl",
                          "sine_wave_horizontal_td", "sine_wave_horizontal_bu",
                          "expanding_circle", "contracting_circle"], {"tooltip": "Shape and motion direction (td/bu = top-down/bottom-up, lr/rl = left-right/right-left)."}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Swap shape and background shades."}),
                "band_thickness_ratio": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Band thickness as a fraction of the frame (moving_band modes)."}),
                "wave_amplitude_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Wave amplitude as a fraction of the frame (sine_wave modes)."}),
                "wave_frequency": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1, "tooltip": "Number of wave periods across the frame (sine_wave modes)."}),
                "wave_thickness_ratio": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01, "tooltip": "Wave line thickness as a fraction of the frame (sine_wave modes)."}),
                "gradient_type": (["none", "all_edges", "trailing_edge"], {"default": "none", "tooltip": "Soft edge style: around all edges, or only behind the direction of motion."}),
                "gradient_width_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Gradient width relative to the shape size. 0 = hard edges."}),
            }
        }

    def generate_mask_animation(self, width, height, total_frames, num_shades, mode, invert_mask,
                                band_thickness_ratio, wave_amplitude_ratio, wave_frequency, wave_thickness_ratio,
                                gradient_type, gradient_width_ratio):
        generator = MaskAnimationGenerator(width, height, total_frames, num_shades)
        if gradient_type != "none" and gradient_width_ratio > 0 and generator.num_shades == 2:
            logging.warning("AnimatedShapeMaskNode: num_shades=2 gives a hard black/white mask; increase num_shades (e.g. 32) to see the gradient.")

        forward = "_td" in mode or "_lr" in mode

        def render(i):
            if mode.startswith("moving_band"):
                return generator.moving_band(i, mode.startswith("moving_band_vertical"), forward, band_thickness_ratio,
                                             invert_mask, gradient_type, gradient_width_ratio)
            if mode.startswith("sine_wave"):
                return generator.sine_wave(i, mode.startswith("sine_wave_vertical"), forward, wave_amplitude_ratio, wave_frequency,
                                           wave_thickness_ratio, invert_mask, gradient_type, gradient_width_ratio)
            return generator.expanding_contracting_circle(i, mode == "expanding_circle", invert_mask, gradient_type, gradient_width_ratio)

        frames = np.empty((total_frames, height, width), dtype=np.float32)
        pbar = comfy.utils.ProgressBar(total_frames)
        for i in range(total_frames):
            frames[i] = render(i)
            pbar.update(1)
        frames /= 255.0
        return (_gray_to_image(frames), num_shades, width, height)
