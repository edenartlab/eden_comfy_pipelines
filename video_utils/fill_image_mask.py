from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils


@dataclass
class FillNodeConfig:
    growth_threshold: float = 0.9
    barrier_jump_power: float = 0.1
    seed: int = 42
    saturation_window: int = 15
    saturation_threshold: float = 1e-5
    # Weights of the individual growth probability maps in the combined map
    weight_lab: float = 0.5
    weight_sam: float = 1.5
    weight_depth: float = 2.0
    weight_canny: float = 0.1
    weight_hed: float = 0.5

    max_steps: int = 5000
    n_frames: int = 100
    processing_resolution: int = 1024
    barrier_override_scaling: float = 0.5
    overlay_alpha: float = 0.4

    circular_bias_strength: float = 0.3  # how much to favor round growth
    jitter_strength: float = 0.2  # random jitter for organic edges
    neighbor_sampling_min: float = 0.6
    neighbor_sampling_max: float = 0.9
    circular_radius_sensitivity: float = 0.3
    organic_bias_range: float = 0.3  # organic multiplier range: (1-range) to (1+range)


_SOBEL_X = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).view(1, 1, 3, 3)


def _sobel_grad(img):
    """Gradient magnitude of single-channel images [B,H,W] -> [B,H,W]."""
    img = img.unsqueeze(1)
    grad_x = F.conv2d(img, _SOBEL_X.to(img.device), padding=1)
    grad_y = F.conv2d(img, _SOBEL_Y.to(img.device), padding=1)
    return torch.sqrt(grad_x**2 + grad_y**2).squeeze(1)


def _normalize(t, eps=1e-8):
    """Min-max normalize each batch item of [B,...] to [0, 1]; constant items become 0."""
    dims = tuple(range(1, t.dim()))
    mn = t.amin(dim=dims, keepdim=True)
    mx = t.amax(dim=dims, keepdim=True)
    rng = mx - mn
    return torch.where(rng < eps, torch.zeros_like(t), (t - mn) / (rng + eps)).float()


def _image_to_lab(img_bhwc):
    """RGB/gray image [B,H,W,C] in 0..1 -> LAB [B,H,W,3] using OpenCV's 8-bit LAB, scaled like the original node."""
    import cv2

    B, H, W, C = img_bhwc.shape
    img = (img_bhwc.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).reshape(B * H, W, C)
    if C == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif C != 3:
        raise ValueError(f"Input image must have 1 or 3 channels, got {C}")
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(B, H, W, 3)
    lab[..., 0] /= 100.0
    lab[..., 1] = (lab[..., 1] + 127.0) / 254.0
    lab[..., 2] = (lab[..., 2] + 127.0) / 254.0
    return torch.from_numpy(lab).to(img_bhwc.device)


def _color_similarity_growth_prob(lab_bhwc, kernel_size_f=0.01):
    """High where the local neighbourhood has uniform color: exp(-10 * variance of the weighted LAB
    distances between each pixel and its kernel_size x kernel_size neighbours)."""
    B, H, W, _ = lab_bhwc.shape
    k = max(int(kernel_size_f * (H + W) / 2), 3)
    k += 1 - k % 2
    p = k // 2
    weights = torch.tensor([0.6, 0.2, 0.2], device=lab_bhwc.device).view(1, 3, 1, 1)
    lab = lab_bhwc.permute(0, 3, 1, 2)
    padded = F.pad(lab, (p, p, p, p), mode='reflect')

    def distances():
        for dy in range(k):
            for dx in range(k):
                diff = (padded[:, :, dy:dy + H, dx:dx + W] - lab) * weights
                yield torch.sqrt(torch.sum(diff**2, dim=1))

    n = k * k
    mean = sum(distances()) / n
    variance = sum((d - mean)**2 for d in distances()) / (n - 1)
    return torch.clamp(torch.exp(-variance * 10.0), 0.0, 1.0)


def _sam_transition_penalty(sam_rgb_bhwc, threshold=0.25):
    """RGB segmentation map [B,H,W,3] -> [B,H,W] binary penalty that is 1 on segment borders."""
    B, H, W, _ = sam_rgb_bhwc.shape
    grads = _sobel_grad(sam_rgb_bhwc.float().permute(0, 3, 1, 2).reshape(B * 3, H, W)).view(B, 3, H, W)
    grad = torch.sqrt(grads[:, 0]**2 + grads[:, 1]**2 + grads[:, 2]**2)
    mn = grad.amin(dim=(1, 2), keepdim=True)
    rng = grad.amax(dim=(1, 2), keepdim=True) - mn
    penalty = torch.where(rng == 0, torch.zeros_like(grad), (grad - mn) / rng)
    return (penalty > threshold).float()


class OrganicFillBatch:
    """Stochastic region growing from seed pixels, steered by a growth probability map built from the image and optional guide maps."""

    def __init__(self, input_image, base_mask, seed_locations, depth=None, canny=None, hed=None, sam=None, config=None, device=None):
        self.cfg = config or FillNodeConfig()
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(self.cfg.seed)

        self.input_image = input_image.to(device)
        self.inside = base_mask.to(device) > 0.5
        B, H, W = self.inside.shape
        self.B, self.H, self.W = B, H, W

        self.depth = depth
        self.canny = canny
        self.hed = hed
        self.sam_rgb = sam

        self.frame_count = 0
        self.fill_history = []
        self.y_grid, self.x_grid = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                                                  torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        self.smooth_kernel = torch.ones(1, 1, 5, 5, device=device) / 25.0

        self._prepare_grow_prob(seed_locations.to(device).float())
        self._place_seeds(seed_locations.to(device).float())

    def _prepare_grow_prob(self, seed_locations):
        cfg = self.cfg
        maps = [_normalize(_color_similarity_growth_prob(_image_to_lab(self.input_image)))]
        weights = [cfg.weight_lab]

        if self.sam_rgb is not None and cfg.weight_sam > 0:
            maps.append(_normalize(1.0 - _sam_transition_penalty(self.sam_rgb)))
            weights.append(cfg.weight_sam)
        if self.depth is not None and cfg.weight_depth > 0:
            depth_grad = _normalize(_sobel_grad(_normalize(self.depth)))
            maps.append(_normalize(torch.exp(-4 * depth_grad)))
            weights.append(cfg.weight_depth)
        if self.canny is not None and cfg.weight_canny > 0:
            maps.append(_normalize(1.0 - _normalize(self.canny.float())))
            weights.append(cfg.weight_canny)
        if self.hed is not None and cfg.weight_hed > 0:
            maps.append(_normalize(1.0 - _normalize(self.hed.float())))
            weights.append(cfg.weight_hed)

        weights_t = torch.tensor(weights, device=self.device).view(-1, 1, 1, 1)
        weighted_sum = torch.sum(torch.stack(maps) * weights_t, dim=0)
        total_weight = torch.sum(weights_t)
        combined = weighted_sum / total_weight if total_weight > 0 else weighted_sum
        self.grow_prob = _normalize(combined)

        # Nearly zero everywhere: boost so the fill can still grow
        if self.grow_prob.max() < 0.05:
            self.grow_prob = torch.clamp(self.grow_prob + 0.1, 0, 1)

    def _place_seeds(self, seed_locations):
        B, H, W = self.B, self.H, self.W
        if torch.sum(seed_locations) > 0:
            filled = seed_locations.clone()
        else:
            filled = torch.zeros((B, H, W), device=self.device, dtype=torch.float32)
            filled[:, H // 2, W // 2] = 1.0
        self.filled_region = filled * self.inside
        # Step at which each pixel got filled (-1 for seeds, int32 max for never), to rebuild any frame later
        self.fill_step = torch.full((B, H, W), torch.iinfo(torch.int32).max, device=self.device, dtype=torch.int32)
        self.fill_step[self.filled_region > 0] = -1

    def _boundary(self):
        neighbours = F.max_pool2d(self.filled_region.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        return (neighbours > 0) & (self.filled_region == 0)

    def _circular_bias(self):
        cfg = self.cfg
        if torch.sum(self.filled_region) == 0:
            return torch.ones(self.B, self.H, self.W, device=self.device)
        maps = []
        for filled_b in self.filled_region:
            if torch.sum(filled_b) == 0:
                maps.append(torch.ones(self.H, self.W, device=self.device))
                continue
            y_coords, x_coords = torch.where(filled_b > 0.5)
            dist = torch.sqrt((self.y_grid - torch.mean(y_coords.float()))**2 + (self.x_grid - torch.mean(x_coords.float()))**2)
            avg_radius = torch.mean(dist[filled_b > 0.5])
            maps.append(torch.exp(-torch.abs(dist - avg_radius) / (avg_radius * cfg.circular_radius_sensitivity)))
        return torch.stack(maps)

    def step(self):
        cfg = self.cfg
        B, H, W = self.B, self.H, self.W
        shape = (B, H, W)
        self.frame_count += 1
        P = self.grow_prob

        potential_growth_base = (self.filled_region == 0) & self.inside & self._boundary()
        circular_bias = self._circular_bias()

        jitter = torch.rand(shape, device=self.device, generator=self.generator)
        jitter = F.conv2d(F.pad(jitter.unsqueeze(1), (2, 2, 2, 2), mode='reflect'), self.smooth_kernel).squeeze(1)

        sampling_rate = cfg.neighbor_sampling_min + (cfg.neighbor_sampling_max - cfg.neighbor_sampling_min) * torch.rand(shape, device=self.device, generator=self.generator)
        neighbor_sampling_mask = torch.rand(shape, device=self.device, generator=self.generator) < sampling_rate

        if cfg.barrier_jump_power == 0.0:
            potential_growth = potential_growth_base & (P > cfg.growth_threshold)
            effective_prob = P
        else:
            alpha = cfg.barrier_jump_power
            organic_boost = circular_bias * cfg.circular_bias_strength + jitter * cfg.jitter_strength
            effective_prob = torch.clamp((1 - alpha) * P + alpha * (P**(1 / 4) + organic_boost * cfg.organic_bias_range), 0.0, 1.0)
            relaxed_threshold = cfg.growth_threshold * (1 - alpha * cfg.barrier_override_scaling)
            potential_growth = potential_growth_base & (P > relaxed_threshold) & neighbor_sampling_mask

        noise = torch.rand(shape, device=self.device, generator=self.generator)
        biased_prob = torch.clamp(effective_prob * (1 - cfg.organic_bias_range + cfg.organic_bias_range * circular_bias), 0.0, 1.0)
        new_growth = potential_growth & (noise < biased_prob)

        self.filled_region[new_growth] = 1.0
        self.fill_step[new_growth] = self.frame_count - 1
        self.fill_history.append(self.fill_ratio())

    def fill_ratio(self):
        mask_area = self.inside.sum(dim=(1, 2)).float()
        filled_area = self.filled_region.sum(dim=(1, 2)).float()
        return torch.where(mask_area > 0, filled_area / mask_area, torch.zeros_like(mask_area))

    def is_complete(self):
        """Done when, for every batch item, the fill ratio saturated or no boundary pixels are left."""
        if len(self.fill_history) < self.cfg.saturation_window:
            return False
        recent = torch.stack(self.fill_history[-self.cfg.saturation_window:])
        saturated = torch.max(torch.abs(recent[1:] - recent[:-1]), dim=0)[0] < self.cfg.saturation_threshold
        no_more_boundary = (self._boundary() & self.inside).sum(dim=(1, 2)) == 0
        return bool(torch.all(saturated | no_more_boundary))


def _to_bhwc(t):
    if t.dim() == 2:
        return t[None, ..., None]
    if t.dim() == 3:
        return t[None] if t.shape[-1] in (1, 3) else t[..., None]
    return t


def _resize_to_processing_resolution(t_bhwc, target_res):
    """Downscale (never upscale) so the longest side is target_res, keeping aspect ratio with even sides."""
    _, h, w, _ = t_bhwc.shape
    if max(h, w) <= target_res:
        return t_bhwc
    scale = target_res / max(h, w)
    new_h = max(int(round(h * scale / 2)) * 2, 2)
    new_w = max(int(round(w * scale / 2)) * 2, 2)
    return F.interpolate(t_bhwc.permute(0, 3, 1, 2), size=(new_h, new_w), mode='bicubic', align_corners=False).permute(0, 2, 3, 1)


def _match_batch_and_size(t_bhwc, B, H, W):
    if t_bhwc.shape[0] != B:
        t_bhwc = t_bhwc[0:1].repeat(B, 1, 1, 1)
    if t_bhwc.shape[1:3] != (H, W):
        t_bhwc = F.interpolate(t_bhwc.permute(0, 3, 1, 2).float(), size=(H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    return t_bhwc


def _prep_mask(mask, B, H, W, device):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] != B:
        mask = mask[0:1].repeat(B, 1, 1)
    if mask.shape[1:3] != (H, W):
        mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
    return mask.to(device).float()


def _prep_gray_map(aux, target_res, B, H, W, device):
    if aux is None:
        return None
    aux = _match_batch_and_size(_resize_to_processing_resolution(_to_bhwc(aux), target_res), B, H, W)
    if aux.shape[-1] == 3:
        aux = 0.299 * aux[..., 0] + 0.587 * aux[..., 1] + 0.114 * aux[..., 2]
    else:
        aux = aux[..., 0]
    return aux.to(device).float()


def _prep_rgb_map(aux, target_res, B, H, W, device):
    if aux is None:
        return None
    aux = _match_batch_and_size(_resize_to_processing_resolution(_to_bhwc(aux), target_res), B, H, W)
    if aux.shape[-1] == 1:
        aux = aux.expand(-1, -1, -1, 3)
    return aux[..., :3].to(device).float()


def _upscale(t_bchw, h, w):
    return F.interpolate(t_bchw, size=(h, w), mode='bicubic', align_corners=False).clamp(0, 1)


class OrganicFillNode:
    DESCRIPTION = "Deprecated: use Organic Fill Animation instead. Grows a mask outward from seed locations, following image color regions and optional SAM/depth/edge guide maps, and returns the growth as an animation."

    @classmethod
    def INPUT_TYPES(cls):
        d = FillNodeConfig()
        return {
            "required": {
                "input_image": ("IMAGE", {"tooltip": "Image whose color regions guide the growth."}),
                "seed_locations": ("MASK", {"tooltip": "Where the fill starts. Empty mask = start at the image center."}),
            },
            "optional": {
                "fill_mask": ("MASK", {"tooltip": "Only grow inside this mask. Defaults to the full image."}),
                "SAM_map": ("IMAGE", {"tooltip": "Flat-color segmentation map; growth avoids crossing segment borders."}),
                "depth_map": ("IMAGE", {"tooltip": "Depth map; growth avoids crossing depth edges."}),
                "canny_map": ("IMAGE", {"tooltip": "Canny edge map; growth avoids crossing edges."}),
                "hed_map": ("IMAGE", {"tooltip": "HED edge map; growth avoids crossing edges."}),
                "n_frames": ("INT", {"default": d.n_frames, "min": 1, "max": 10000, "tooltip": "Number of evenly spaced frames in the output animation."}),
                "max_steps": ("INT", {"default": d.max_steps, "min": 1, "max": 10000, "tooltip": "Maximum number of growth steps."}),
                "growth_threshold": ("FLOAT", {"default": d.growth_threshold, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Pixels need a growth probability above this (relaxed by barrier_jump_power) to be filled."}),
                "barrier_jump_power": ("FLOAT", {"default": d.barrier_jump_power, "min": 0, "max": 1, "step": 0.01, "tooltip": "How easily growth crosses low-probability barriers. 0 = strictly follow the probability map."}),
                "weight_lab": ("FLOAT", {"default": d.weight_lab, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Weight of the image color-similarity map."}),
                "weight_sam": ("FLOAT", {"default": d.weight_sam, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Weight of the SAM segment-border map."}),
                "weight_depth": ("FLOAT", {"default": d.weight_depth, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Weight of the depth-edge map."}),
                "weight_canny": ("FLOAT", {"default": d.weight_canny, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Weight of the canny edge map."}),
                "weight_hed": ("FLOAT", {"default": d.weight_hed, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Weight of the HED edge map."}),
                "seed": ("INT", {"default": d.seed, "min": 0, "max": 2147483647, "tooltip": "Random seed for the stochastic growth."}),
                "processing_resolution": ("INT", {"default": d.processing_resolution, "min": 256, "max": 4096, "step": 64, "tooltip": "Longest side used for the simulation; outputs are scaled back to the input size."}),
                "saturation_window": ("INT", {"default": d.saturation_window, "min": 5, "max": 100, "tooltip": "Stop when the fill ratio barely changed over this many steps."}),
                "saturation_threshold": ("FLOAT", {"default": d.saturation_threshold, "min": 1e-6, "max": 1e-4, "step": 1e-5, "tooltip": "Fill ratio change per step below which growth counts as saturated."}),
                "circular_bias_strength": ("FLOAT", {"default": d.circular_bias_strength, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much to favor round, blob-like growth."}),
                "jitter_strength": ("FLOAT", {"default": d.jitter_strength, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Random jitter for more organic edges."}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("final_mask", "frames_preview", "grow_prob_map", "overlayed_fill_preview")
    OUTPUT_TOOLTIPS = ("Final filled mask.", "Fill animation frames (first batch item).", "Growth probability map.", "Fill animation overlaid in red on the input image.")
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Deprecated"

    def execute(self, input_image, seed_locations, fill_mask=None, SAM_map=None, depth_map=None, canny_map=None, hed_map=None, **params):
        cfg = FillNodeConfig(**params)
        device = comfy.model_management.get_torch_device()
        B, orig_H, orig_W, _ = input_image.shape

        input_image = _resize_to_processing_resolution(input_image, cfg.processing_resolution).clamp(0, 1)
        B, H, W, _ = input_image.shape

        seed_locations = _prep_mask(seed_locations, B, H, W, device)
        base_mask = _prep_mask(fill_mask, B, H, W, device) if fill_mask is not None else torch.ones((B, H, W), dtype=torch.float32, device=device)

        res = cfg.processing_resolution
        fill = OrganicFillBatch(
            input_image=input_image,
            base_mask=base_mask,
            seed_locations=seed_locations,
            depth=_prep_gray_map(depth_map, res, B, H, W, device),
            canny=_prep_gray_map(canny_map, res, B, H, W, device),
            hed=_prep_gray_map(hed_map, res, B, H, W, device),
            sam=_prep_rgb_map(SAM_map, res, B, H, W, device),
            config=cfg,
            device=device,
        )

        pbar = comfy.utils.ProgressBar(cfg.max_steps)
        for _ in range(cfg.max_steps):
            comfy.model_management.throw_exception_if_processing_interrupted()
            fill.step()
            pbar.update(1)
            if fill.is_complete():
                break

        # Frame k is the fill state after step k; pick n_frames of them evenly spaced
        n_steps, n_frames = fill.frame_count, cfg.n_frames
        if n_steps < n_frames:
            indices = list(range(n_steps)) + [n_steps - 1] * (n_frames - n_steps)
        elif n_frames == 1:
            indices = [n_steps - 1]
        else:
            indices = [int(i * (n_steps - 1) / (n_frames - 1)) for i in range(n_frames)]
        indices = torch.tensor(indices, device=device, dtype=torch.int32).view(-1, 1, 1)
        frames = fill.filled_region[0] * (fill.fill_step[0] <= indices)  # [N,H,W]

        # Red overlay of the fill on the input image, built on the intermediate device to spare VRAM
        out = comfy.model_management.intermediate_device()
        image = input_image[0].to(out)
        if image.shape[-1] == 1:
            image = image.expand(-1, -1, 3)
        image = image[..., :3]
        frames_out = frames.to(out)
        mask_rgb = torch.zeros(frames_out.shape + (3,), device=out)
        mask_rgb[..., 0] = frames_out
        mask_alpha = (frames_out * cfg.overlay_alpha).unsqueeze(-1)
        overlay = image * (1 - mask_alpha) + mask_rgb * mask_alpha

        final_mask, grow_prob, frames = fill.filled_region, fill.grow_prob, frames.unsqueeze(1)
        if (H, W) != (orig_H, orig_W):
            final_mask = _upscale(final_mask.unsqueeze(1), orig_H, orig_W).squeeze(1)
            grow_prob = _upscale(grow_prob.unsqueeze(1), orig_H, orig_W).squeeze(1)
            frames = _upscale(frames, orig_H, orig_W)
            overlay = _upscale(overlay.permute(0, 3, 1, 2), orig_H, orig_W).permute(0, 2, 3, 1)

        frames_preview = frames.squeeze(1).to(out).unsqueeze(-1).expand(-1, -1, -1, 3)
        return final_mask.to(out), frames_preview, grow_prob.to(out), overlay
