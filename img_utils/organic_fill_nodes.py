import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import comfy.model_management


class StartPosition(Enum):
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    TOP_CENTER = "top_center"
    BOTTOM_CENTER = "bottom_center"
    LEFT_CENTER = "left_center"
    RIGHT_CENTER = "right_center"
    RANDOM = "random"


@dataclass
class FillConfig:
    target_size: int = 264
    threshold_value: int = 128
    seed_radius: int = 5
    noise_range: Tuple[float, float] = (0.0, 1.2)
    growth_threshold: float = 0.6
    stability_threshold: int = 20
    saturation_window: int = 15
    saturation_threshold: float = 0.001
    fps: float = 20.0
    starting_position: StartPosition = StartPosition.RANDOM
    position_randomness: float = 1.0
    island_connection_radius: float = 0.005
    active_region_padding: int = 3
    invert_input: bool = False
    num_seeds: int = 1
    color_channel: str = "luminance"
    loop: bool = True
    hold_final_frame_fraction: float = 0.25


class OrganicFill:
    """Grows a filled region from seed points into a binary shape mask, one noisy dilation step per frame.

    With `seed` set, all randomness comes from private generators (reproducible, global RNG untouched);
    otherwise the global torch / python RNGs are used."""

    def __init__(self, config: FillConfig, seed=None, rng=random):
        self.config = config
        self.device = comfy.model_management.get_torch_device()
        self.rng = rng
        self.fill_history = []
        self.cpu_gen = self.dev_gen = None
        if seed is not None:
            self.cpu_gen = torch.Generator().manual_seed(seed)
            self.dev_gen = self.cpu_gen if self.device.type == "cpu" else torch.Generator(self.device).manual_seed(seed)

    def _process_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self.config.island_connection_radius == 0:
            return mask.to(device=self.device, dtype=torch.float32)
        import cv2
        mask_np = mask.cpu().numpy().astype(np.uint8)
        kernel_size = max(3, int(self.config.island_connection_radius * max(mask_np.shape)))
        kernel_size += (kernel_size % 2 == 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.erode(cv2.dilate(mask_np, kernel, iterations=2), kernel, iterations=1)
        return torch.from_numpy(np.maximum(mask_np, closed)).to(device=self.device, dtype=torch.float32)

    def load_from_tensor(self, image_tensor: torch.Tensor) -> None:
        """Initialize from a ComfyUI IMAGE tensor (B,H,W,3); only the first image is used."""
        import cv2
        img = image_tensor[0]
        channel = self.config.color_channel
        if channel == "random":
            gray = img[:, :, self.rng.randint(0, 2)]
        elif channel in ("red", "green", "blue"):
            gray = img[:, :, ("red", "green", "blue").index(channel)]
        else:
            gray = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        img_np = (gray.cpu().numpy() * 255).astype(np.uint8)

        h, w = img_np.shape[:2]
        max_dim = self.config.target_size
        if w >= h:
            new_w, new_h = max_dim, int(h * (max_dim / w))
        else:
            new_h, new_w = max_dim, int(w * (max_dim / h))
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if self.config.invert_input:
            img_np = 255 - img_np

        self.mask = self._process_mask(torch.tensor(img_np < self.config.threshold_value))
        self.mask_sum = self.mask.sum()
        self.height, self.width = self.mask.shape

        self.grid = torch.zeros_like(self.mask)
        self.activity_counter = torch.zeros_like(self.mask, dtype=torch.int32)
        self.active_mask = torch.ones_like(self.mask, dtype=torch.bool)
        self._place_seed()

    def _random_index(self, n):
        return torch.randint(n, (1,), generator=self.cpu_gen).to(self.device)

    def _place_seed(self) -> None:
        h, w = self.mask.shape
        valid_coords = torch.where(self.mask > 0)
        if len(valid_coords[0]) == 0:
            return

        cfg = self.config
        if cfg.num_seeds == 1:
            position_filters = {
                StartPosition.TOP_LEFT: lambda y, x: (y < h // 2) & (x < w // 2),
                StartPosition.TOP_RIGHT: lambda y, x: (y < h // 2) & (x >= w // 2),
                StartPosition.BOTTOM_LEFT: lambda y, x: (y >= h // 2) & (x < w // 2),
                StartPosition.BOTTOM_RIGHT: lambda y, x: (y >= h // 2) & (x >= w // 2),
                StartPosition.CENTER: lambda y, x: (y >= h // 3) & (y < 2 * h // 3) & (x >= w // 3) & (x < 2 * w // 3),
                StartPosition.TOP_CENTER: lambda y, x: (y < h // 3) & (x >= w // 3) & (x < 2 * w // 3),
                StartPosition.BOTTOM_CENTER: lambda y, x: (y >= 2 * h // 3) & (x >= w // 3) & (x < 2 * w // 3),
                StartPosition.LEFT_CENTER: lambda y, x: (x < w // 3) & (y >= h // 3) & (y < 2 * h // 3),
                StartPosition.RIGHT_CENTER: lambda y, x: (x >= 2 * w // 3) & (y >= h // 3) & (y < 2 * h // 3),
                StartPosition.RANDOM: lambda y, x: torch.ones_like(y, dtype=torch.bool),
            }
            valid_mask = position_filters[cfg.starting_position](*valid_coords)
            if not torch.any(valid_mask):
                valid_mask = position_filters[StartPosition.CENTER](*valid_coords)
                if not torch.any(valid_mask):
                    valid_mask = position_filters[StartPosition.RANDOM](*valid_coords)

            valid_points = torch.stack([coord[valid_mask] for coord in valid_coords])
            if cfg.position_randomness == 0.0:
                seed_y, seed_x = valid_points[0].float().mean(), valid_points[1].float().mean()
            elif cfg.position_randomness == 1.0:
                seed_y, seed_x = valid_points[:, self._random_index(valid_points.shape[1])]
            else:
                center_y, center_x = valid_points[0].float().mean(), valid_points[1].float().mean()
                random_y, random_x = valid_points[:, self._random_index(valid_points.shape[1])].float()
                seed_y = center_y + cfg.position_randomness * (random_y - center_y)
                seed_x = center_x + cfg.position_randomness * (random_x - center_x)
            seed_positions = [(seed_x, seed_y)]
        else:
            valid_points = torch.stack(valid_coords)
            center_y, center_x = h / 2.0, w / 2.0
            radius = min(h, w) / 4.0
            seed_positions = []
            for i in range(cfg.num_seeds):
                angle = 2 * np.pi * i / cfg.num_seeds
                sym_x = center_x + radius * np.cos(angle)
                sym_y = center_y + radius * np.sin(angle)
                if cfg.position_randomness == 0.0:
                    seed_positions.append((max(0, min(w - 1, sym_x)), max(0, min(h - 1, sym_y))))
                    continue
                idx = self._random_index(valid_points.shape[1])
                if cfg.position_randomness == 1.0:
                    sy, sx = valid_points[:, idx]
                else:
                    rand_y, rand_x = valid_points[:, idx].float()
                    sx = sym_x + cfg.position_randomness * (rand_x - sym_x)
                    sy = sym_y + cfg.position_randomness * (rand_y - sym_y)
                seed_positions.append((float(sx), float(sy)))

        y, x = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing="ij",
        )
        for sx, sy in seed_positions:
            seed = (y - sy) ** 2 + (x - sx) ** 2 <= cfg.seed_radius ** 2
            self.grid[seed & (self.mask > 0)] = 1.0

    def _update_activity(self, new_growth: torch.Tensor) -> None:
        self.activity_counter = torch.where(new_growth, 0, self.activity_counter + 1)
        recent_activity = (self.activity_counter < self.config.stability_threshold).float()
        pad = self.config.active_region_padding
        padded_active = F.max_pool2d(recent_activity[None, None], 2 * pad + 1, stride=1, padding=pad)[0, 0] > 0
        self.active_mask = padded_active & (self.grid < 1) & (self.mask > 0)

    def step(self) -> None:
        # grid is binary, so "any filled 8-neighbour" of an empty pixel == 3x3 max-pool > 0
        has_filled_neighbor = F.max_pool2d(self.grid[None, None], 3, stride=1, padding=1)[0, 0] > 0
        boundary = (self.grid == 0) & (self.mask > 0) & has_filled_neighbor & self.active_mask

        low, high = self.config.noise_range
        noise = torch.rand(self.grid.shape, generator=self.dev_gen, device=self.device) * (high - low) + low

        new_growth = boundary & (noise > self.config.growth_threshold)
        self.grid[new_growth] = 1
        self._update_activity(new_growth)
        self.fill_history.append(float((self.grid * self.mask).sum() / self.mask_sum))

    def is_complete(self) -> bool:
        window = self.config.saturation_window
        if len(self.fill_history) < window:
            return False
        recent_change = max(abs(self.fill_history[i] - self.fill_history[i - 1]) for i in range(-window + 1, 0))
        return recent_change < self.config.saturation_threshold and not torch.any(
            (self.grid == 0) & self.active_mask & (self.mask > 0)
        )

    def get_frame(self) -> np.ndarray:
        return (self.grid * 255).to(torch.uint8).cpu().numpy()


def resample_frames(frames: List[np.ndarray], target_n_frames: int) -> List[np.ndarray]:
    """Linearly resample a frame sequence to target_n_frames, blending neighbouring frames."""
    if len(frames) == target_n_frames:
        return frames
    if target_n_frames == 1:
        return [frames[len(frames) // 2]]

    resampled = []
    n_input = len(frames)
    for i in range(target_n_frames):
        source_pos = i * (n_input - 1) / (target_n_frames - 1)
        frame_idx = int(source_pos)
        blend = source_pos - frame_idx
        if frame_idx >= n_input - 1:
            resampled.append(frames[-1])
        elif blend < 1e-6:
            resampled.append(frames[frame_idx])
        else:
            f1 = frames[frame_idx].astype(np.float32)
            f2 = frames[frame_idx + 1].astype(np.float32)
            resampled.append((f1 * (1 - blend) + f2 * blend).astype(np.uint8))
    return resampled


def grow_starting_frames(starting_frame: np.ndarray, num_frames: int, np_rng) -> List[np.ndarray]:
    """Fade in the first fill frame by revealing a growing random subset of its white pixels."""
    white_pixels = np.where(starting_frame > 0)
    total = len(white_pixels[0])
    frames = []
    for fi in range(num_frames):
        frame = np.zeros_like(starting_frame)
        n_pixels = int(((fi + 1) / num_frames) ** 2 * total)
        if n_pixels > 0:
            indices = np_rng.choice(total, n_pixels, replace=False)
            frame[white_pixels[0][indices], white_pixels[1][indices]] = 255
        frames.append(frame)
    return frames


def generate_reverse_frames(forward_frames: List[np.ndarray]) -> List[np.ndarray]:
    """Un-fill: starting from the last frame, remove pixels in the order they appeared."""
    current = forward_frames[-1].copy()
    reverse = [current.copy()]
    for i in range(1, len(forward_frames)):
        current[forward_frames[i] > forward_frames[i - 1]] = 0
        reverse.append(current.copy())
    return reverse


def _to_image(frames: np.ndarray) -> torch.Tensor:
    """(N, H, W) uint8 -> ComfyUI IMAGE (N, H, W, 3) float32 in 0..1."""
    gray = torch.from_numpy(frames).float() / 255.0
    return gray.unsqueeze(-1).repeat(1, 1, 1, 3)


def run_organic_fill(image_tensor: torch.Tensor, config: FillConfig, target_n_seconds: float, seed=None, rng=random):
    """Run the organic fill algorithm and return frames as a ComfyUI IMAGE tensor (N,H,W,3)."""
    import comfy.utils

    fill = OrganicFill(config, seed=seed, rng=rng)
    fill.load_from_tensor(image_tensor)
    target_n_frames = max(1, int(target_n_seconds * config.fps))

    if fill.mask_sum == 0:  # nothing to fill: every frame would be black
        return _to_image(np.zeros((target_n_frames, fill.height, fill.width), dtype=np.uint8))

    max_steps = 10000
    pbar = comfy.utils.ProgressBar(max_steps)
    forward_frames = []
    while not fill.is_complete() and len(forward_frames) < max_steps:
        fill.step()
        forward_frames.append(fill.get_frame())
        pbar.update(1)
    pbar.update_absolute(max_steps)

    np_rng = np.random if seed is None else np.random.RandomState(seed)
    complete_forward = grow_starting_frames(forward_frames[0], 10, np_rng) + forward_frames
    all_frames = list(complete_forward)

    if config.loop:
        hold = int(config.hold_final_frame_fraction * len(complete_forward))
        all_frames.extend([complete_forward[-1]] * hold)

        reverse = generate_reverse_frames(complete_forward)
        all_frames.extend(reverse)

        hold = int(config.hold_final_frame_fraction * len(forward_frames)) // 2
        all_frames.extend([reverse[-1]] * hold)

    return _to_image(np.stack(resample_frames(all_frames, target_n_frames)))


class Eden_OrganicFillAnimation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Shape image; dark pixels (below threshold_value) are the region that gets filled."}),
                "target_seconds": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 120.0, "step": 0.5, "tooltip": "Output duration; frames = target_seconds * fps."}),
                "target_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8, "tooltip": "Longest side of the simulation (and output) resolution."}),
                "threshold_value": ("INT", {"default": 128, "min": 1, "max": 254, "step": 1, "tooltip": "Pixels darker than this (0-255) belong to the fillable shape."}),
                "seed_radius": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1, "tooltip": "Radius in pixels of each starting seed."}),
                "growth_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "A boundary pixel grows when its noise exceeds this. Higher = slower, more ragged growth."}),
                "noise_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Lower bound of the per-pixel growth noise."}),
                "noise_high": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 3.0, "step": 0.1, "tooltip": "Upper bound of the per-pixel growth noise."}),
                "starting_position": (
                    ["center", "top_left", "top_right", "bottom_left", "bottom_right",
                     "top_center", "bottom_center", "left_center", "right_center", "random"],
                    {"default": "center", "tooltip": "Region of the shape where the (single) seed is placed."},
                ),
                "position_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "0 = seed at the region's center, 1 = at a random pixel of the region."}),
                "island_connection_radius": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 0.05, "step": 0.0001, "tooltip": "Closes small gaps between shape islands (fraction of image size) so growth can cross them."}),
                "active_region_padding": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1, "tooltip": "Pixels around recently grown areas that stay eligible for growth."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0, "tooltip": "Frames per second of the output."}),
                "num_seeds": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "tooltip": "Number of seeds; more than one are placed on a circle around the center."}),
                "invert_input": ("BOOLEAN", {"default": False, "tooltip": "Fill the light areas instead of the dark ones."}),
                "loop": ("BOOLEAN", {"default": True, "tooltip": "Append a hold and an un-fill so the clip loops."}),
                "hold_final_frame_fraction": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "When looping: how long to hold the filled frame, relative to the fill duration."}),
                "color_channel": (["luminance", "red", "green", "blue", "random"], {"default": "luminance", "tooltip": "Which channel of the image is thresholded."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "Eden 🌱/Mask"
    DESCRIPTION = "Creates an organic, noisy growth animation that fills the dark shape of the input image, as a grayscale mask video."
    OUTPUT_TOOLTIPS = ("Fill animation frames (white = filled).",)

    def generate(
        self, image, target_seconds, target_size, threshold_value, seed_radius,
        growth_threshold, noise_low, noise_high, starting_position,
        position_randomness, island_connection_radius, active_region_padding,
        fps, num_seeds, invert_input, loop, hold_final_frame_fraction, color_channel,
    ):
        config = FillConfig(
            target_size=target_size,
            threshold_value=threshold_value,
            seed_radius=seed_radius,
            noise_range=(noise_low, noise_high),
            growth_threshold=growth_threshold,
            stability_threshold=20,
            saturation_window=15,
            saturation_threshold=0.0001,
            fps=fps,
            starting_position=StartPosition(starting_position),
            position_randomness=position_randomness,
            island_connection_radius=island_connection_radius,
            active_region_padding=active_region_padding,
            invert_input=invert_input,
            num_seeds=num_seeds,
            color_channel=color_channel,
            loop=loop,
            hold_final_frame_fraction=hold_final_frame_fraction,
        )
        return (run_organic_fill(image, config, target_seconds),)


class Eden_GradientBorderMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1200, "min": 8, "max": 8192, "step": 8, "tooltip": "Output width in pixels."}),
                "height": ("INT", {"default": 800, "min": 8, "max": 8192, "step": 8, "tooltip": "Output height in pixels."}),
                "border_fraction": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.001, "tooltip": "Border width as a fraction of the longest side."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Eden 🌱/Mask"
    DESCRIPTION = "Generates a white image whose edges fade linearly to black over the border width."
    OUTPUT_TOOLTIPS = ("Gradient border image.",)

    def generate(self, width, height, border_fraction):
        border = int(border_fraction * max(width, height))
        y, x = np.ogrid[:height, :width]
        dist_from_edge = np.minimum(np.minimum(y, height - 1 - y), np.minimum(x, width - 1 - x))
        img = 255 - np.clip((border - dist_from_edge) * 255 / max(border, 1), 0, 255).astype(np.uint8)
        return (_to_image(img[None]),)


class Eden_OrganicFillRandom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Shape image to fill (see Organic Fill Animation)."}),
                "target_seconds": ("FLOAT", {"default": 24.0, "min": 0.5, "max": 120.0, "step": 0.5, "tooltip": "Output duration; frames = target_seconds * fps."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Seed for both the sampled parameters and the growth noise; same seed = same animation."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0, "tooltip": "Frames per second of the output."}),
                "loop": ("BOOLEAN", {"default": True, "tooltip": "Append a hold and an un-fill so the clip loops."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("frames", "params_info",)
    FUNCTION = "generate"
    CATEGORY = "Eden 🌱/Mask"
    DESCRIPTION = "Organic fill animation with all growth parameters randomly sampled from the seed, for quick variations."
    OUTPUT_TOOLTIPS = ("Fill animation frames (white = filled).", "The sampled parameters.")

    def generate(self, image, target_seconds, seed, fps, loop):
        rng = random.Random(seed)
        target_size = rng.choice([512, 512, 768])
        threshold_value = rng.randint(80, 192)
        seed_radius = rng.randint(3, 10)
        noise_range = rng.choice([
            (0.0, 0.8), (0.0, 1.2), (0.0, 2.0),
            (0.0, 1.0), (0.2, 1.0), (0.1, 1.5),
        ])
        growth_threshold = rng.uniform(0.3, 0.8)
        stability_threshold = rng.randint(11, 30)
        saturation_window = rng.randint(12, 25)
        saturation_threshold = rng.uniform(0.0001, 0.01)
        starting_position = rng.choice(list(StartPosition))
        position_randomness = rng.uniform(0.0, 0.75)
        island_connection_radius = rng.uniform(0.0, 0.01)
        active_region_padding = rng.randint(1, 8)
        invert_input = rng.choice([True, True, True, False])
        num_seeds = rng.randint(1, 2)
        color_channel = rng.choice(["red", "random", "luminance", "random", "green", "blue"])
        hold_final_frame_fraction = rng.uniform(0.8, 1.0)

        config = FillConfig(
            target_size=target_size,
            threshold_value=threshold_value,
            seed_radius=seed_radius,
            noise_range=noise_range,
            growth_threshold=growth_threshold,
            stability_threshold=stability_threshold,
            saturation_window=saturation_window,
            saturation_threshold=saturation_threshold,
            fps=fps,
            starting_position=starting_position,
            position_randomness=position_randomness,
            island_connection_radius=island_connection_radius,
            active_region_padding=active_region_padding,
            invert_input=invert_input,
            num_seeds=num_seeds,
            color_channel=color_channel,
            loop=loop,
            hold_final_frame_fraction=hold_final_frame_fraction,
        )
        frames = run_organic_fill(image, config, target_seconds, seed=seed, rng=rng)

        params_info = (
            f"target_size={target_size}, threshold={threshold_value}, "
            f"seed_radius={seed_radius}, growth_threshold={growth_threshold:.2f}, "
            f"noise_range={noise_range}, position={starting_position.value}, "
            f"pos_randomness={position_randomness:.2f}, island_conn={island_connection_radius:.4f}, "
            f"invert={invert_input}, num_seeds={num_seeds}, channel={color_channel}, "
            f"hold_frac={hold_final_frame_fraction:.2f}"
        )
        return (frames, params_info)
