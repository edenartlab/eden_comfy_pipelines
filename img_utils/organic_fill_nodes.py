import torch
import numpy as np
import cv2
import random
import sys
from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum


# ────────────────────────────────────────────────────────────────────────────────
#  CORE ALGORITHM (from fill_polygon.py)
# ────────────────────────────────────────────────────────────────────────────────

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
    device: str = "cpu"
    loop: bool = True
    hold_final_frame_fraction: float = 0.25


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OrganicFill:
    def __init__(self, config: FillConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.fill_history = []
        self.frame_count = 0
        self.seed_location = None

    def _process_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self.config.island_connection_radius == 0:
            return mask.to(device=self.device, dtype=torch.float32)
        mask_np = mask.cpu().numpy().astype(np.uint8)
        kernel_size = max(3, int(self.config.island_connection_radius * max(mask_np.shape)))
        kernel_size += (kernel_size % 2 == 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return torch.from_numpy(
            np.maximum(
                mask_np,
                cv2.erode(cv2.dilate(mask_np, kernel, iterations=2), kernel, iterations=1),
            )
        ).to(device=self.device, dtype=torch.float32)

    def load_from_tensor(self, image_tensor: torch.Tensor) -> None:
        """Initialize from a ComfyUI IMAGE tensor (B,H,W,3) or grayscale (H,W)."""
        if image_tensor.dim() == 4:
            img = image_tensor[0]  # take first in batch
            # Convert to grayscale via luminance
            if self.config.color_channel == "red":
                gray = img[:, :, 0]
            elif self.config.color_channel == "green":
                gray = img[:, :, 1]
            elif self.config.color_channel == "blue":
                gray = img[:, :, 2]
            elif self.config.color_channel == "random":
                gray = img[:, :, random.randint(0, 2)]
            else:  # luminance
                gray = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
            img_np = (gray.cpu().numpy() * 255).astype(np.uint8)
        elif image_tensor.dim() == 3:
            img_np = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

        # Resize to target_size
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
        self.height, self.width = self.mask.shape

        self.grid = torch.zeros_like(self.mask)
        self.activity_counter = torch.zeros_like(self.mask, dtype=torch.int32)
        self.active_mask = torch.ones_like(self.mask, dtype=torch.bool)
        self._place_seed()

    def _place_seed(self) -> None:
        h, w = self.mask.shape
        valid_coords = torch.where(self.mask > 0)

        if len(valid_coords[0]) == 0:
            self.seed_location = (0.5, 0.5)
            return

        if self.config.num_seeds == 1:
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
            valid_mask = position_filters[self.config.starting_position](valid_coords[0], valid_coords[1])
            if not torch.any(valid_mask):
                valid_mask = position_filters[StartPosition.CENTER](valid_coords[0], valid_coords[1])
                if not torch.any(valid_mask):
                    valid_mask = position_filters[StartPosition.RANDOM](valid_coords[0], valid_coords[1])

            valid_points = torch.stack([coord[valid_mask] for coord in valid_coords])
            if self.config.position_randomness == 0.0:
                seed_y, seed_x = valid_points[0].float().mean(), valid_points[1].float().mean()
            elif self.config.position_randomness == 1.0:
                idx = torch.randint(valid_points.shape[1], (1,))
                seed_y, seed_x = valid_points[:, idx]
            else:
                center_y, center_x = valid_points[0].float().mean(), valid_points[1].float().mean()
                idx = torch.randint(valid_points.shape[1], (1,))
                random_y, random_x = valid_points[:, idx].float()
                seed_y = center_y + self.config.position_randomness * (random_y - center_y)
                seed_x = center_x + self.config.position_randomness * (random_x - center_x)
            seed_positions = [(seed_x, seed_y)]
        else:
            valid_points = torch.stack([coord for coord in valid_coords])
            if self.config.position_randomness == 0.0:
                center_y, center_x = h / 2.0, w / 2.0
                seed_positions = []
                for i in range(self.config.num_seeds):
                    angle = 2 * np.pi * i / self.config.num_seeds
                    radius = min(h, w) / 4.0
                    sx = max(0, min(w - 1, center_x + radius * np.cos(angle)))
                    sy = max(0, min(h - 1, center_y + radius * np.sin(angle)))
                    seed_positions.append((sx, sy))
            elif self.config.position_randomness == 1.0:
                seed_positions = []
                for _ in range(self.config.num_seeds):
                    idx = torch.randint(valid_points.shape[1], (1,))
                    sy, sx = valid_points[:, idx]
                    seed_positions.append((float(sx), float(sy)))
            else:
                center_y, center_x = h / 2.0, w / 2.0
                seed_positions = []
                for i in range(self.config.num_seeds):
                    angle = 2 * np.pi * i / self.config.num_seeds
                    radius = min(h, w) / 4.0
                    sym_x = center_x + radius * np.cos(angle)
                    sym_y = center_y + radius * np.sin(angle)
                    idx = torch.randint(valid_points.shape[1], (1,))
                    rand_y, rand_x = valid_points[:, idx].float()
                    sx = sym_x + self.config.position_randomness * (rand_x - sym_x)
                    sy = sym_y + self.config.position_randomness * (rand_y - sym_y)
                    seed_positions.append((float(sx), float(sy)))

        y, x = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing="ij",
        )
        for sx, sy in seed_positions:
            seed = (y - sy) ** 2 + (x - sx) ** 2 <= self.config.seed_radius ** 2
            self.grid[seed & (self.mask > 0)] = 1.0

        if seed_positions:
            first_sx, first_sy = seed_positions[0]
            self.seed_location = (float(first_sx) / self.width, float(first_sy) / self.height)

    def _update_activity(self, new_growth: torch.Tensor) -> None:
        self.activity_counter[new_growth] = 0
        self.activity_counter[~new_growth] += 1
        recent_activity = self.activity_counter < self.config.stability_threshold
        recent_np = recent_activity.cpu().numpy()
        kernel = np.ones((2 * self.config.active_region_padding + 1,) * 2, np.uint8)
        padded_active = cv2.dilate(recent_np.astype(np.uint8), kernel)
        self.active_mask = (
            torch.from_numpy(padded_active).to(self.device).bool()
            & (self.grid < 1)
            & (self.mask > 0)
        )

    def step(self) -> None:
        self.frame_count += 1
        padded = torch.nn.functional.pad(self.grid, (1, 1, 1, 1))
        neighbors = torch.nn.functional.unfold(
            padded.unsqueeze(0).unsqueeze(0), kernel_size=3, padding=0
        ).squeeze().view(9, self.height, self.width)

        boundary = (
            (self.grid == 0)
            & (self.mask > 0)
            & ((neighbors.sum(0) - neighbors[4]) > 0)
            & self.active_mask
        )

        noise = torch.rand_like(self.grid) * (
            self.config.noise_range[1] - self.config.noise_range[0]
        ) + self.config.noise_range[0]

        new_growth = boundary & (noise > self.config.growth_threshold)
        self.grid[new_growth] = 1
        self._update_activity(new_growth)
        self.fill_history.append(self.compute_fill_ratio())

    def compute_fill_ratio(self) -> float:
        return float((self.grid * self.mask).sum() / self.mask.sum())

    def is_complete(self) -> bool:
        if len(self.fill_history) < self.config.saturation_window:
            return False
        recent_change = max(
            abs(self.fill_history[i] - self.fill_history[i - 1])
            for i in range(-self.config.saturation_window + 1, 0)
        )
        return recent_change < self.config.saturation_threshold and not torch.any(
            (self.grid == 0) & self.active_mask & (self.mask > 0)
        )

    def get_frame(self) -> np.ndarray:
        return (self.grid.cpu().numpy() * 255).astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────

def resample_frames(frames: List[np.ndarray], target_n_frames: int) -> List[np.ndarray]:
    if not frames or target_n_frames <= 0:
        return []
    if len(frames) == target_n_frames:
        return frames.copy()
    if target_n_frames == 1:
        return [frames[len(frames) // 2]]

    resampled = []
    n_input = len(frames)
    for i in range(target_n_frames):
        source_pos = i * (n_input - 1) / (target_n_frames - 1)
        frame_idx = int(source_pos)
        blend = source_pos - frame_idx
        if frame_idx >= n_input - 1:
            resampled.append(frames[-1].copy())
        elif blend < 1e-6:
            resampled.append(frames[frame_idx].copy())
        else:
            f1 = frames[frame_idx].astype(np.float32)
            f2 = frames[frame_idx + 1].astype(np.float32)
            resampled.append((f1 * (1 - blend) + f2 * blend).astype(np.uint8))
    return resampled


def grow_starting_frames(starting_frame: np.ndarray, num_frames: int) -> List[np.ndarray]:
    if num_frames <= 0:
        return []
    white_pixels = np.where(starting_frame > 0)
    total = len(white_pixels[0])
    if total == 0:
        return [np.zeros_like(starting_frame) for _ in range(num_frames)]

    frames = []
    for fi in range(num_frames):
        frame = np.zeros_like(starting_frame)
        progress = ((fi + 1) / num_frames) ** 2
        n_pixels = int(progress * total)
        if n_pixels > 0:
            indices = np.random.choice(total, n_pixels, replace=False)
            frame[white_pixels[0][indices], white_pixels[1][indices]] = 255
        frames.append(frame)
    return frames


def generate_reverse_frames(forward_frames: List[np.ndarray]) -> List[np.ndarray]:
    if not forward_frames:
        return []
    reverse = []
    current = forward_frames[-1].copy()
    reverse.append(current.copy())
    for i in range(1, len(forward_frames)):
        newly_white = forward_frames[i] > forward_frames[i - 1]
        current[newly_white] = 0
        reverse.append(current.copy())
    return reverse


def run_organic_fill(image_tensor: torch.Tensor, config: FillConfig, target_n_seconds: float):
    """Run the organic fill algorithm and return frames as a ComfyUI IMAGE tensor (N,H,W,3)."""
    fill = OrganicFill(config)
    fill.load_from_tensor(image_tensor)

    forward_frames = []
    max_steps = 10000
    step = 0
    while not fill.is_complete() and step < max_steps:
        fill.step()
        forward_frames.append(fill.get_frame().copy())
        step += 1

    if not forward_frames:
        # Return single black frame
        h, w = fill.height, fill.width
        blank = np.zeros((1, h, w, 3), dtype=np.float32)
        return torch.from_numpy(blank)

    # Grow starting frames
    starting = grow_starting_frames(forward_frames[0], 10)
    complete_forward = starting + forward_frames
    all_frames = complete_forward.copy()

    if config.loop:
        if config.hold_final_frame_fraction > 0:
            hold = int(config.hold_final_frame_fraction * len(complete_forward))
            if hold > 0:
                final = complete_forward[-1].copy()
                all_frames.extend([final] * hold)

        reverse = generate_reverse_frames(complete_forward)
        all_frames.extend(reverse)

        if config.hold_final_frame_fraction > 0:
            hold = int(config.hold_final_frame_fraction * len(forward_frames)) // 2
            if hold > 0:
                all_frames.extend([reverse[-1].copy()] * hold)

    target_n_frames = max(1, int(target_n_seconds * config.fps))
    all_frames = resample_frames(all_frames, target_n_frames)

    # Convert to ComfyUI IMAGE format (N, H, W, 3) float [0,1]
    stacked = np.stack(all_frames)  # (N, H, W) uint8
    stacked_f = stacked.astype(np.float32) / 255.0
    stacked_rgb = np.stack([stacked_f, stacked_f, stacked_f], axis=-1)  # (N, H, W, 3)
    return torch.from_numpy(stacked_rgb)


# ────────────────────────────────────────────────────────────────────────────────
#  NODE 1: Organic Fill Animation
# ────────────────────────────────────────────────────────────────────────────────

class Eden_OrganicFillAnimation:
    """Creates an animated organic growth fill pattern from an input image mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_seconds": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 120.0, "step": 0.5}),
                "target_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "threshold_value": ("INT", {"default": 128, "min": 1, "max": 254, "step": 1}),
                "seed_radius": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "growth_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "noise_high": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 3.0, "step": 0.1}),
                "starting_position": (
                    ["center", "top_left", "top_right", "bottom_left", "bottom_right",
                     "top_center", "bottom_center", "left_center", "right_center", "random"],
                    {"default": "center"},
                ),
                "position_randomness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "island_connection_radius": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 0.05, "step": 0.0001}),
                "active_region_padding": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "num_seeds": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "invert_input": ("BOOLEAN", {"default": False}),
                "loop": ("BOOLEAN", {"default": True}),
                "hold_final_frame_fraction": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
                "color_channel": (["luminance", "red", "green", "blue", "random"], {"default": "luminance"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "Eden 🌱/animation"

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
            device=_get_device(),
            loop=loop,
            hold_final_frame_fraction=hold_final_frame_fraction,
        )
        frames = run_organic_fill(image, config, target_seconds)
        return (frames,)


# ────────────────────────────────────────────────────────────────────────────────
#  NODE 2: Gradient Border Mask
# ────────────────────────────────────────────────────────────────────────────────

class Eden_GradientBorderMask:
    """Generates a gradient border image (black edges fading to white center)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1200, "min": 8, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 800, "min": 8, "max": 8192, "step": 8}),
                "border_fraction": ("FLOAT", {"default": 0.03, "min": 0.001, "max": 0.5, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Eden 🌱/masks"

    def generate(self, width, height, border_fraction):
        border = int(border_fraction * max(width, height))
        y, x = np.ogrid[:height, :width]
        dist_from_edge = np.minimum(
            np.minimum(y, height - 1 - y),
            np.minimum(x, width - 1 - x),
        )
        img = 255 - np.clip((border - dist_from_edge) * 255 / max(border, 1), 0, 255).astype(np.uint8)
        # Convert to ComfyUI IMAGE format (1, H, W, 3) float [0,1]
        img_f = img.astype(np.float32) / 255.0
        img_rgb = np.stack([img_f, img_f, img_f], axis=-1)
        return (torch.from_numpy(img_rgb).unsqueeze(0),)


# ────────────────────────────────────────────────────────────────────────────────
#  NODE 3: Random Organic Fill Variations
# ────────────────────────────────────────────────────────────────────────────────

class Eden_OrganicFillRandom:
    """Generates an organic fill animation with randomly sampled parameters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_seconds": ("FLOAT", {"default": 24.0, "min": 0.5, "max": 120.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "loop": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("frames", "params_info",)
    FUNCTION = "generate"
    CATEGORY = "Eden 🌱/animation"

    def generate(self, image, target_seconds, seed, fps, loop):
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        # Sample random parameters (ranges from many_fill_polygon.py)
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
            device=_get_device(),
            loop=loop,
            hold_final_frame_fraction=hold_final_frame_fraction,
        )

        # Use the numpy rng for reproducibility within the fill algorithm
        old_state = np.random.get_state()
        np.random.seed(seed)
        torch.manual_seed(seed)
        frames = run_organic_fill(image, config, target_seconds)
        np.random.set_state(old_state)

        params_info = (
            f"target_size={target_size}, threshold={threshold_value}, "
            f"seed_radius={seed_radius}, growth_threshold={growth_threshold:.2f}, "
            f"noise_range={noise_range}, position={starting_position.value}, "
            f"pos_randomness={position_randomness:.2f}, island_conn={island_connection_radius:.4f}, "
            f"invert={invert_input}, num_seeds={num_seeds}, channel={color_channel}, "
            f"hold_frac={hold_final_frame_fraction:.2f}"
        )
        return (frames, params_info)
