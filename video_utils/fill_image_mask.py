import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image
from torchvision.transforms import Resize

# Hacky way to get this to work as module and standalone script (for testing)
try:
    from fill_utils import *
except:
    from .fill_utils import *

# ────────────────────────────────────────────────────────────────────────────────
#  CONFIG & ENUMS
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class FillNodeConfig:
    growth_threshold: float = 0.9  # Lower threshold to allow more growth
    barrier_jump_power: float = 0.1  # Higher barrier jump to overcome high probability areas
    seed: int = 42  # Random seed for reproducibility
    saturation_window: int = 15
    saturation_threshold: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Growth probability map weights for weighted combination
    weight_lab: float = 0.5
    weight_sam: float = 1.5
    weight_depth: float = 2.0
    weight_canny: float = 0.1
    weight_hed: float = 0.5

    max_steps: int = 5000
    n_frames: int = 100
    processing_resolution: int = 1024
    # Additional algorithm constants
    uniform_high_prob: float = 0.8
    barrier_override_scaling: float = 0.5
    overlay_alpha: float = 0.4

# ────────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────

_SOBEL_X = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).view(1,1,3,3)
_SOBEL_Y = torch.tensor([[1., 2.,  1.], [0., 0.,  0.], [-1., -2., -1.]]).view(1,1,3,3)

def _sobel_grad(img: torch.Tensor) -> torch.Tensor:
    """Return gradient magnitude of a single‑channel image [B,H,W] -> [B,H,W]."""
    if img.dim() != 3:
        raise ValueError(f"Input to _sobel_grad must be [B,H,W], got {img.shape}")
    img_b1hw = img.unsqueeze(1) # Add channel dim: [B,1,H,W]
    grad_x = F.conv2d(img_b1hw, _SOBEL_X.to(img.device), padding=1)
    grad_y = F.conv2d(img_b1hw, _SOBEL_Y.to(img.device), padding=1)
    mag = torch.sqrt(grad_x**2 + grad_y**2)
    return mag.squeeze(1)  # [B,H,W]

def _image_bhwc_to_lab_bhwc(img_bhwc: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor [B,H,W,C] (0-1 float) to full LAB colorspace [B,H,W,3]."""
    B, H, W, C = img_bhwc.shape
    lab_batch = []
    device = img_bhwc.device
    img_bhwc_cpu_numpy = (img_bhwc.cpu().numpy() * 255).astype(np.uint8)

    for b in range(B):
        img_hwc = img_bhwc_cpu_numpy[b]
        # Ensure 3 channels for RGB conversion
        if C == 1:
            img_hwc = cv2.cvtColor(img_hwc, cv2.COLOR_GRAY2RGB)
        elif C != 3:
             raise ValueError(f"Input image must have 1 or 3 channels, got {C}")
        
        lab_hwc = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2LAB)
        # Normalize LAB channels: L* (0-100) -> (0-1), a* and b* (-127 to 127) -> (0-1)
        lab_hwc_norm = lab_hwc.astype(np.float32)
        lab_hwc_norm[..., 0] /= 100.0  # L* channel: 0-100 -> 0-1
        lab_hwc_norm[..., 1] = (lab_hwc_norm[..., 1] + 127.0) / 254.0  # a* channel: -127 to 127 -> 0-1
        lab_hwc_norm[..., 2] = (lab_hwc_norm[..., 2] + 127.0) / 254.0  # b* channel: -127 to 127 -> 0-1
        
        lab_batch.append(torch.from_numpy(lab_hwc_norm))

    return torch.stack(lab_batch).to(device) # [B,H,W,3]

def _compute_color_similarity_growth_prob(lab_bhwc: torch.Tensor, kernel_size_f: float = 0.01) -> torch.Tensor:
    """
    Compute growth probability based on perceptual color similarity in LAB space.
    Areas with similar colors get high growth probability.
    
    Args:
        lab_bhwc: [B,H,W,3] LAB color space image (normalized 0-1)
        kernel_size: Size of local neighborhood for color similarity computation
    
    Returns:
        color_grow_prob: [B,H,W] growth probability map (0-1)
    """
    B, H, W, C = lab_bhwc.shape
    device = lab_bhwc.device

    kernel_size = int(kernel_size_f * (H + W) / 2)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Use adaptive weights for LAB channels based on perceptual importance
    # L* is most perceptually important, a* and b* contribute to color similarity
    lab_weights = torch.tensor([0.6, 0.2, 0.2], device=device).view(1, 1, 1, 3)  # [1,1,1,3]
    
    color_similarity_maps = []
    
    for b in range(B):
        lab_hwc = lab_bhwc[b]  # [H,W,3]
        
        # Compute local color variance using a sliding window approach
        # Pad the image for convolution
        pad_size = kernel_size // 2
        lab_padded = F.pad(lab_hwc.permute(2, 0, 1), (pad_size, pad_size, pad_size, pad_size), mode='reflect')  # [3,H+pad,W+pad]
        
        # Unfold to get local neighborhoods for each pixel
        lab_unfolded = F.unfold(lab_padded.unsqueeze(0), kernel_size=kernel_size, stride=1)  # [1, 3*kernel_size^2, H*W]
        lab_unfolded = lab_unfolded.squeeze(0).view(3, kernel_size*kernel_size, H, W)  # [3, K^2, H, W]
        lab_unfolded = lab_unfolded.permute(2, 3, 0, 1)  # [H, W, 3, K^2]
        
        # Get the center pixel for each neighborhood
        center_idx = kernel_size * kernel_size // 2
        center_colors = lab_unfolded[..., center_idx]  # [H, W, 3]
        
        # Compute weighted color distances from center to all neighbors
        center_expanded = center_colors.unsqueeze(-1)  # [H, W, 3, 1]
        color_diffs = lab_unfolded - center_expanded  # [H, W, 3, K^2]
        
        # Apply perceptual weights and compute L2 distance
        weighted_diffs = color_diffs * lab_weights.squeeze(0).squeeze(0).unsqueeze(-1)  # [H, W, 3, K^2]
        distances = torch.sqrt(torch.sum(weighted_diffs**2, dim=2))  # [H, W, K^2]
        
        # Compute local color variance (lower variance = more similar colors)
        color_variance = torch.var(distances, dim=-1)  # [H, W]
        
        # Convert variance to similarity (high variance = low similarity)
        # Use exponential decay to create smooth transitions
        color_similarity = torch.exp(-color_variance * 10.0)  # Scale factor controls sensitivity
        
        color_similarity_maps.append(color_similarity)
    
    color_grow_prob = torch.stack(color_similarity_maps)  # [B, H, W]
    return torch.clamp(color_grow_prob, 0.0, 1.0)

def _sam_rgb_to_color_transition_penalty(sam_rgb_bhwc: torch.Tensor, threshold_value: float = 0.25) -> torch.Tensor:
    """
    Convert RGB SAM segmentation map to color transition penalty map.
    Args:
        sam_rgb_bhwc: [B,H,W,C] RGB segmentation map where each flat color represents a segment
        threshold_value: Threshold for detecting color transitions (default from config)
    Returns:
        penalty_bhw: [B,H,W] penalty map where transitions between colors get high penalty (0-1)
    """
    B, H, W, C = sam_rgb_bhwc.shape
    device = sam_rgb_bhwc.device
    
    if C != 3:
        raise ValueError(f"SAM map must be RGB (3 channels), got {C}")
    
    # Convert to device and ensure float
    sam_rgb = sam_rgb_bhwc.to(device).float()
    
    # Compute RGB gradients for each channel separately
    penalty_maps = []
    for b in range(B):
        rgb_b = sam_rgb[b]  # [H,W,C]
        
        # Compute gradient magnitude for each RGB channel separately
        r_grad = _sobel_grad(rgb_b[..., 0].unsqueeze(0)).squeeze(0)  # [H,W]
        g_grad = _sobel_grad(rgb_b[..., 1].unsqueeze(0)).squeeze(0)  # [H,W]
        b_grad = _sobel_grad(rgb_b[..., 2].unsqueeze(0)).squeeze(0)  # [H,W]
        
        # Combine RGB gradients - use L2 norm across channels
        combined_grad = torch.sqrt(r_grad**2 + g_grad**2 + b_grad**2)
        
        # Normalize the penalty to 0-1:
        if combined_grad.max() - combined_grad.min() == 0:
            penalty = torch.zeros_like(combined_grad)
        else:
            penalty = (combined_grad - combined_grad.min()) / (combined_grad.max() - combined_grad.min())

        # Threshold the penalty:
        penalty = torch.where(penalty > threshold_value, torch.ones_like(penalty), torch.zeros_like(penalty))
        penalty_maps.append(penalty)
    
    penalty_bhw = torch.stack(penalty_maps)  # [B,H,W]
    return torch.clamp(penalty_bhw, 0.0, 1.0)

# ────────────────────────────────────────────────────────────────────────────────
#  CORE ORGANIC‑FILL IMPLEMENTATION (BATCH‑AWARE)
# ────────────────────────────────────────────────────────────────────────────────

class OrganicFillBatch:
    """Performs confidence‑aware organic fill on a batch of images."""
    def __init__(self,
                 input_image: torch.Tensor, # [B,H,W,C] float 0-1 RGB/Gray
                 base_mask: torch.Tensor,   # [B,H,W] float 0/1 (Note: Currently always full image)
                 seed_locations: torch.Tensor, # [B,H,W] float 0/1 binary mask for seed placement
                 depth: Optional[torch.Tensor] = None,  # [B,H,W] float 0‑1
                 canny: Optional[torch.Tensor] = None,  # [B,H,W] float/bool
                 hed: Optional[torch.Tensor] = None,  # [B,H,W] float/bool
                 sam: Optional[torch.Tensor] = None,    # [B,H,W,C] RGB float 0‑1 (segmentation map)
                 config: FillNodeConfig = None):

        self.cfg = config or FillNodeConfig()
        self.device = torch.device(self.cfg.device)

        # Set random seed for reproducibility
        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed_all(self.cfg.seed)
        print(f"Set random seed to {self.cfg.seed} for reproducible results")

        self.input_image = input_image.to(self.device)
        self.mask = base_mask.to(self.device).float()
        # Don't normalize the mask - it's already a valid binary mask (0/1)
        # self.mask = normalize_tensor(self.mask) # threshold in/out hardcoded at 0.5

        self.seed_locations = seed_locations.to(self.device).float()  # 1 for seed locations, 0 otherwise
        B, H, W = self.mask.shape
        self.B, self.H, self.W = B, H, W

        # Optional channels
        self.depth = depth.to(self.device) if depth is not None else None
        self.canny = canny.to(self.device) if canny is not None else None
        self.hed   = hed.to(self.device)   if hed is not None else None
        self.sam_rgb = sam.to(self.device) if sam is not None else None  # Store RGB SAM for color transitions
        
        # Grids
        self.filled_region = torch.zeros_like(self.mask)
        self.frame_count = 0
        self.fill_history: List[torch.Tensor] = []

        # Precompute sobel kernels on device
        global _SOBEL_X, _SOBEL_Y
        _SOBEL_X = _SOBEL_X.to(self.device)
        _SOBEL_Y = _SOBEL_Y.to(self.device)

        print(f"Preparing growth probability fields on {self.device}...")
        self._prepare_grow_prob_fields()
        self._place_seeds()

    # ────────────────────────────────────────────────────────────────────────
    #  GROWTH PROBABILITY FIELD
    # ────────────────────────────────────────────────────────────────────────
    def _prepare_grow_prob_fields(self, save_maps: bool = True):
        """Compute independent growth probability maps for each input and combine with weighted sum."""
        B, H, W = self.B, self.H, self.W
        device = self.device

        # Check which maps are available
        has_sam   = self.sam_rgb is not None
        has_depth = self.depth   is not None
        has_canny = self.canny   is not None
        has_hed   = self.hed     is not None

        print(f"Computing growth probability from available maps: LAB={True}, SAM={has_sam}, Depth={has_depth}, Canny={has_canny}, HED={has_hed}")

        # Initialize list to store individual growth probability maps and their weights
        grow_prob_maps = []
        weights = []

        # ── LAB Perceptual Color Similarity Growth Probability (always available) ──
        print("Computing LAB perceptual color similarity growth probability...")
        lab_bhwc = _image_bhwc_to_lab_bhwc(self.input_image)  # [B,H,W,3]
        lab_grow_prob = _compute_color_similarity_growth_prob(lab_bhwc)  # [B,H,W]
        lab_grow_prob = normalize_tensor(lab_grow_prob)  # [B,H,W] 0-1
        
        if save_maps:
            comfy_tensor_to_pil(lab_grow_prob.unsqueeze(-1)).save('grow_prob_lab.png')
        grow_prob_maps.append(lab_grow_prob)
        weights.append(self.cfg.weight_lab)

        # ── SAM RGB Segmentation Growth Probability (if available) ──
        if has_sam and self.cfg.weight_sam > 0:
            # SAM is RGB - compute color transition penalty of shape: [B,H,W]
            sam_penalty = _sam_rgb_to_color_transition_penalty(self.sam_rgb)

            # Convert penalty to growth probability (high penalty -> low growth probability)
            sam_grow_prob = 1.0 - sam_penalty
            sam_grow_prob = normalize_tensor(sam_grow_prob)  # [B,H,W] 0-1

            if save_maps:
                comfy_tensor_to_pil(sam_grow_prob.unsqueeze(-1)).save('grow_prob_sam.png')
            grow_prob_maps.append(sam_grow_prob)
            weights.append(self.cfg.weight_sam)

        if has_depth and self.cfg.weight_depth > 0:
            d_norm = normalize_tensor(self.depth)
            depth_grad_mag = _sobel_grad(d_norm)  # [B,H,W]
            depth_grad_mag = normalize_tensor(depth_grad_mag)

            # High depth gradient -> low growth probability
            depth_grow_prob = torch.exp(-4 * depth_grad_mag)
            depth_grow_prob = normalize_tensor(depth_grow_prob)  # [B,H,W] 0-1
            if save_maps:
                comfy_tensor_to_pil(depth_grow_prob.unsqueeze(-1)).save('grow_prob_depth.png')
            grow_prob_maps.append(depth_grow_prob)
            weights.append(self.cfg.weight_depth)

        # ── Canny Edge Growth Probability (if available) ──
        if has_canny and self.cfg.weight_canny > 0:
            c_norm = normalize_tensor(self.canny.float())
            # High edge strength -> low growth probability
            canny_grow_prob = 1.0 - c_norm
            canny_grow_prob = normalize_tensor(canny_grow_prob)  # [B,H,W] 0-1
            if save_maps:
                comfy_tensor_to_pil(canny_grow_prob.unsqueeze(-1)).save('grow_prob_canny.png')
            grow_prob_maps.append(canny_grow_prob)
            weights.append(self.cfg.weight_canny)

        # ── HED Edge Growth Probability (if available) ──
        if has_hed and self.cfg.weight_hed > 0:
            h_norm = normalize_tensor(self.hed.float())
            # High edge strength -> low growth probability
            hed_grow_prob = 1.0 - h_norm
            hed_grow_prob = normalize_tensor(hed_grow_prob)  # [B,H,W] 0-1
            if save_maps:
                comfy_tensor_to_pil(hed_grow_prob.unsqueeze(-1)).save('grow_prob_hed.png')
            grow_prob_maps.append(hed_grow_prob)
            weights.append(self.cfg.weight_hed)

        # ── Combine Growth Probability Maps with Weighted Sum ──
        if not grow_prob_maps:
            # Fallback: use uniform growth probability if no maps available
            print("Warning: No growth probability maps available, using uniform growth probability.")
            combined_grow_prob = torch.ones((B, H, W), device=device)
        else:
            print(f"Combining {len(grow_prob_maps)} growth probability maps with weighted sum...")
            # Stack maps and weights
            maps_tensor = torch.stack(grow_prob_maps, dim=0)  # [N, B, H, W]
            weights_tensor = torch.tensor(weights, device=device).view(-1, 1, 1, 1)  # [N, 1, 1, 1]
            
            # Weighted sum
            weighted_sum = torch.sum(maps_tensor * weights_tensor, dim=0)  # [B, H, W]
            total_weight = torch.sum(weights_tensor)
            
            # Normalize by total weight
            combined_grow_prob = weighted_sum / total_weight if total_weight > 0 else weighted_sum
            
        # Final normalization of [B,H,W] tensor:
        self.grow_prob = normalize_tensor(combined_grow_prob)
        
        # Debug: Check if growth probability is too low everywhere
        prob_min, prob_max, prob_mean = self.grow_prob.min().item(), self.grow_prob.max().item(), self.grow_prob.mean().item()
        print(f"Growth probability computed: range=[{prob_min:.3f}, {prob_max:.3f}], mean={prob_mean:.3f}")
        
        # More conservative fallback - only boost if very low maximum
        if prob_max < 0.05:  # Much lower threshold for boost
            print("Warning: Growth probability is extremely low everywhere. Boosting values to enable growth.")
            self.grow_prob = torch.clamp(self.grow_prob + 0.1, 0, 1)  # Smaller boost
            prob_min, prob_max, prob_mean = self.grow_prob.min().item(), self.grow_prob.max().item(), self.grow_prob.mean().item()
            print(f"Boosted growth probability: range=[{prob_min:.3f}, {prob_max:.3f}], mean={prob_mean:.3f}")

        if save_maps:
            comfy_tensor_to_pil(self.grow_prob.unsqueeze(-1)).save('grow_prob.png')

    # ────────────────────────────────────────────────────────────────────────
    #  SEED PLACEMENT
    # ────────────────────────────────────────────────────────────────────────
    def _place_seeds(self):
        B,H,W = self.B,self.H,self.W

        if torch.sum(self.seed_locations) > 0:
            self.filled_region = self.seed_locations.clone()
        else:
            self.filled_region = torch.zeros((B,H,W), device=self.device, dtype=torch.float32)
            self.filled_region[:,H//2,W//2] = 1.0

        # Apply the mask to the grid
        self.filled_region = self.filled_region * (self.mask > 0.5)
        
        # Debug: Print total seeded area per batch
        for b in range(B):
            total_seeded = torch.sum(self.filled_region[b]).item()
            total_mask_area = torch.sum(self.mask[b] > 0.5).item()
            print(f"Batch {b}: Seeded {total_seeded} pixels ({100*total_seeded/total_mask_area:.2f}% of fillable area)")

    # ────────────────────────────────────────────────────────────────────────
    #  CORE STEP
    # ────────────────────────────────────────────────────────────────────────
    def step(self):
        self.frame_count += 1
        B,H,W = self.B,self.H,self.W

        # Find neighbours of currently filled cells
        padded_grid = F.pad(self.filled_region.unsqueeze(1), (1,1,1,1), mode='constant', value=0) # Pad [B,H,W] -> [B,1,H+2,W+2]
        neighbours = F.max_pool2d(padded_grid, kernel_size=3, stride=1, padding=0) # Neighbours include self
        # Identify boundary cells: not filled, but have a filled neighbour, and are within the mask
        has_filled_neighbour = (neighbours.squeeze(1) > 0) & (self.filled_region == 0)

        P = self.grow_prob

        # DEBUG: Track key metrics
        total_filled = torch.sum(self.filled_region).item()
        total_unfilled = torch.sum((self.filled_region == 0) & (self.mask > 0.5)).item()
        total_with_neighbor = torch.sum(has_filled_neighbour).item()
        
        if self.frame_count % 10 == 0 or self.frame_count < 5:  # Print every 10 steps or first 5 steps
            prob_min, prob_max, prob_mean = P.min().item(), P.max().item(), P.mean().item()
            print(f"Step {self.frame_count}: filled={total_filled}, unfilled={total_unfilled}, with_neighbor={total_with_neighbor}")
            print(f"  Growth prob: min={prob_min:.4f}, max={prob_max:.4f}, mean={prob_mean:.4f}")

        # --- Determine potential growth cells ---
        # Must be:
        # 1. Not already filled (grid == 0)
        # 2. Inside the main mask (mask > 0)
        # 3. Have a filled neighbour (has_filled_neighbour)
        potential_growth_base = (self.filled_region == 0) & (self.mask > 0.5) & has_filled_neighbour

        # --- Barrier Jumping Logic ---
        # Convert barrier_jump_power (0-1) to effective noise parameters
        # 0: Respect probability map completely (only grow where P > threshold)
        # 0.5: Moderate barrier jumping ability
        # 1.0: Almost ignore probability map (grow everywhere with high probability)
        
        # Calculate effective growth probability based on barrier jumping strength
        if self.cfg.barrier_jump_power == 0.0:
            # No barrier jumping - must respect probability map completely
            relaxed_threshold = self.cfg.growth_threshold
            potential_growth = potential_growth_base & (P > relaxed_threshold)
            effective_prob = P
        else:
            # Allow barrier jumping with increasing strength
            # Use exponential scaling to make the effect more dramatic at higher values
            
            # Mix the original probability with a barrier-override probability
            # At low jump_power: mostly use original P
            # At high jump_power: mostly use a uniform high probability
            alpha = self.cfg.barrier_jump_power
            #effective_prob = (1 - alpha) * P + alpha * self.cfg.uniform_high_prob
            effective_prob = (1 - alpha) * P + alpha * P**(1/4)
            
            # For potential growth, also relax the threshold requirement based on jump strength
            relaxed_threshold = self.cfg.growth_threshold * (1 - alpha * self.cfg.barrier_override_scaling)  # Can reduce threshold by up to barrier_override_scaling
            potential_growth = potential_growth_base & ((P > relaxed_threshold))

        # DEBUG: Track potential growth areas
        total_potential_base = torch.sum(potential_growth_base).item()
        total_potential_final = torch.sum(potential_growth).item()
        total_above_threshold = torch.sum((self.filled_region == 0) & (self.mask > 0.5) & (P > relaxed_threshold)).item()

        # --- Stochastic Growth ---
        # Generate random noise for stochastic decision making
        noise = torch.rand_like(self.filled_region, device=self.device)
        
        # New growth happens where potential exists AND noise < effective_probability
        new_growth = potential_growth & (noise < effective_prob)
        total_new_growth = torch.sum(new_growth).item()

        # DEBUG: Print detailed step info
        if self.frame_count % 10 == 0 or self.frame_count < 5 or total_new_growth == 0:
            eff_prob_min, eff_prob_max, eff_prob_mean = effective_prob.min().item(), effective_prob.max().item(), effective_prob.mean().item()
            print(f"  Potential: base={total_potential_base}, final={total_potential_final}, above_thresh={total_above_threshold}")
            print(f"  Effective prob: min={eff_prob_min:.4f}, max={eff_prob_max:.4f}, mean={eff_prob_mean:.4f}")
            print(f"  Threshold: {relaxed_threshold:.4f}, New growth: {total_new_growth}")
            if total_new_growth == 0 and total_potential_final > 0:
                print(f"  WARNING: No growth despite {total_potential_final} potential pixels!")

        # Update grid
        self.filled_region[new_growth] = 1.0

        # Calculate fill ratio and track history
        fill_ratio = self.fill_ratio()
        self.fill_history.append(fill_ratio.clone()) # Store fill ratio per batch item

        # Return total boundary pixels for progress reporting
        boundary_count = torch.sum(has_filled_neighbour & (self.filled_region == 0) & (self.mask > 0.5)).item()
        avg_fill_ratio = fill_ratio.mean().item()
        return boundary_count, avg_fill_ratio

    # ────────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────────────────────────
    def fill_ratio(self):
        """Calculate fill ratio per batch item using actual mask area."""
        # Calculate actual mask area per batch item
        mask_area = torch.sum(self.mask > 0.5, dim=(1,2)).float()  # [B] - actual fillable pixels per batch
        filled_area = self.filled_region.sum(dim=(1,2)).float()  # [B] - filled pixels per batch
        # Avoid division by zero if mask area is zero for some batch items
        ratio = torch.where(mask_area > 0, filled_area / mask_area, torch.zeros_like(mask_area, device=self.device))
        return ratio

    def is_complete(self):
        """Check completion based on fill saturation and active pixels per batch item."""
        B = self.B
        # Check 1: Minimum number of steps reached for saturation check
        if len(self.fill_history) < self.cfg.saturation_window:
            return False

        # Check 2: Fill ratio saturation (per batch item)
        # Stack recent history: List of [B] tensors -> Tensor [Window, B]
        recent_history = torch.stack(self.fill_history[-self.cfg.saturation_window:])
        # Calculate max absolute change over the window for each batch item: [B]
        max_change = torch.max(torch.abs(recent_history[1:] - recent_history[:-1]), dim=0)[0]
        saturated = max_change < self.cfg.saturation_threshold

        # Check 3: No more boundary pixels available to fill within the mask (per batch item)
        # Calculate boundary pixels for each batch item: unfilled pixels with filled neighbors
        padded_grid = F.pad(self.filled_region.unsqueeze(1), (1,1,1,1), mode='constant', value=0)
        neighbours = F.max_pool2d(padded_grid, kernel_size=3, stride=1, padding=0)
        has_filled_neighbour_batch = (neighbours.squeeze(1) > 0) & (self.filled_region == 0)
        boundary_pixels = has_filled_neighbour_batch & (self.mask > 0.5)
        no_more_boundary = boundary_pixels.sum(dim=(1,2)) == 0  # [B] - True if no boundary pixels for that batch

        # Complete if saturated OR no more boundary pixels for that batch item
        # Use logical OR element-wise for the batch (either condition should stop)
        is_done = saturated | no_more_boundary

        # DEBUG: Print completion status details
        if self.frame_count % 10 == 0 or torch.any(no_more_boundary) or torch.any(saturated):
            boundary_count_per_batch = boundary_pixels.sum(dim=(1,2))
            print(f"  Completion check: saturation_window={len(self.fill_history)}/{self.cfg.saturation_window}")
            print(f"  Max changes: {max_change.cpu().numpy()}, threshold={self.cfg.saturation_threshold}")
            print(f"  Saturated: {saturated.cpu().numpy()}, No boundary: {no_more_boundary.cpu().numpy()}")
            print(f"  Boundary pixels per batch: {boundary_count_per_batch.cpu().numpy()}")
            print(f"  Is done: {is_done.cpu().numpy()}")

        # The whole process is complete if *all* batch items are done
        return torch.all(is_done)

    def get_frame(self):
        return self.filled_region.clone()  # [B,H,W] float 0/1

# ────────────────────────────────────────────────────────────────────────────────
#  COMFYUI NODE DEFINITION
# ────────────────────────────────────────────────────────────────────────────────

class OrganicFillNode:
    """ComfyUI node wrapping the confidence‑aware Organic Fill algorithm."""

    @classmethod
    def INPUT_TYPES(cls):
        # Get default values from FillNodeConfig
        default_config = FillNodeConfig()
        
        return {
            "required": {
                "input_image": ("IMAGE", {}),  # BHWC RGB or Gray (float 0-1)
                "seed_locations": ("MASK", {}),  # BHW binary mask for seed placement (float 0-1)
            },
            "optional": {
                "fill_mask": ("MASK", {}),     # Optional: BHW mask defining fillable regions (float 0-1)
                "SAM_map": ("IMAGE", {}),      # Optional: BHWC/HWC Probability Map (float 0-1)
                "depth_map": ("IMAGE", {}),    # Optional: BHWC/HWC Depth Map (float 0-1)
                "canny_map": ("IMAGE", {}),    # Optional: BHWC/HWC Canny Edges (float 0-1)
                "hed_map": ("IMAGE", {}),      # Optional: BHWC/HWC HED Edges (float 0-1)
                "n_frames": ("INT", {"default": default_config.n_frames, "min": 1, "max": 10000}),
                "max_steps": ("INT", {"default": default_config.max_steps, "min": 1, "max": 10000}),
                "growth_threshold": ("FLOAT", {"default": default_config.growth_threshold, "min": 0.0, "max": 1.0, "step": 0.01}),
                "barrier_jump_power": ("FLOAT", {"default": default_config.barrier_jump_power, "min": 0, "max": 1, "step": 0.01}),
                "weight_lab": ("FLOAT", {"default": default_config.weight_lab, "min": 0.0, "max": 5.0, "step": 0.01}),
                "weight_sam": ("FLOAT", {"default": default_config.weight_sam, "min": 0.0, "max": 5.0, "step": 0.01}),
                "weight_depth": ("FLOAT", {"default": default_config.weight_depth, "min": 0.0, "max": 5.0, "step": 0.01}),
                "weight_canny": ("FLOAT", {"default": default_config.weight_canny, "min": 0.0, "max": 5.0, "step": 0.01}),
                "weight_hed": ("FLOAT", {"default": default_config.weight_hed, "min": 0.0, "max": 5.0, "step": 0.01}),
                "seed": ("INT", {"default": default_config.seed, "min": 0, "max": 2147483647}),
                "processing_resolution": ("INT", {"default": default_config.processing_resolution, "min": 256, "max": 4096, "step": 64}),
                "saturation_window": ("INT", {"default": default_config.saturation_window, "min": 5, "max": 100}),
                "saturation_threshold": ("FLOAT", {"default": default_config.saturation_threshold, "min": 1e-6, "max": 1e-4, "step": 1e-5}),
            }
        }

    # Output is a single mask (last frame) in ComfyUI MASK format (B, H, W)
    # and a batch of frames (N, H, W) - requires custom handling downstream if used
    # and the growth probability map (B, H, W) in MASK format
    # and overlayed fill preview (N, H, W, C) showing mask animation on input image
    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("final_mask", "frames_preview", "grow_prob_map", "overlayed_fill_preview")

    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Experimental"

    # ────────────────────────────────────────────────────────────────────────
    def execute(self,
                input_image: torch.Tensor, # BHWC, float 0-1
                seed_locations: torch.Tensor, # BHW binary mask for seed placement (float 0-1)
                fill_mask: Optional[torch.Tensor] = None, # BHW mask defining fillable regions (float 0-1)
                SAM_map: Optional[torch.Tensor] = None, # BHWC/HWC
                depth_map: Optional[torch.Tensor] = None, # BHWC/HWC
                canny_map: Optional[torch.Tensor] = None, # BHWC/HWC
                hed_map: Optional[torch.Tensor] = None, # BHWC/HWC
                n_frames: Optional[int] = None,
                max_steps: Optional[int] = None,
                growth_threshold: Optional[float] = None,
                barrier_jump_power: Optional[float] = None,
                weight_lab: Optional[float] = None,
                weight_sam: Optional[float] = None,
                weight_depth: Optional[float] = None,
                weight_canny: Optional[float] = None,
                weight_hed: Optional[float] = None,
                seed: Optional[int] = None,
                processing_resolution: Optional[int] = None,
                saturation_window: Optional[int] = None,
                saturation_threshold: Optional[float] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # Return mask BHW, frames NBHWC, grow_prob BHW, overlayed_fill_preview NHWC

        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create base config with defaults, then override with provided values
        default_config = FillNodeConfig()
        config_kwargs = {
            'device': device,
        }
        
        # Override defaults with any provided values
        if n_frames is not None:
            config_kwargs['n_frames'] = n_frames
        else:
            n_frames = default_config.n_frames
            
        if max_steps is not None:
            config_kwargs['max_steps'] = max_steps
        else:
            max_steps = default_config.max_steps
            
        if processing_resolution is not None:
            config_kwargs['processing_resolution'] = processing_resolution
        else:
            processing_resolution = default_config.processing_resolution
            
        # Add all the optional parameters if provided
        param_mapping = {
            'growth_threshold': growth_threshold,
            'barrier_jump_power': barrier_jump_power,
            'weight_lab': weight_lab,
            'weight_sam': weight_sam,
            'weight_depth': weight_depth,
            'weight_canny': weight_canny,
            'weight_hed': weight_hed,
            'seed': seed,
            'saturation_window': saturation_window,
            'saturation_threshold': saturation_threshold,
        }
        
        for key, value in param_mapping.items():
            if value is not None:
                config_kwargs[key] = value
        
        cfg = FillNodeConfig(**config_kwargs)
        
        print(f"OrganicFill starting on {device} with max {max_steps} steps")
        print(f"Input image dimensions: {input_image.shape}")

        # --- Input Validation and Preparation ---
        if input_image.dim() != 4:
            raise ValueError(f"Expected input_image with shape [B, H, W, C], got {input_image.shape}")
        B, H, W, C = input_image.shape
        orig_H, orig_W = H, W  # Store original dimensions for final output scaling
        
        # --- Resolution Scaling Logic ---
        def _resize_to_processing_resolution(tensor_bhwc: torch.Tensor, target_res: int) -> torch.Tensor:
            """Resize tensor to processing resolution maintaining aspect ratio and rounding to even integers."""
            b, h, w, c = tensor_bhwc.shape
            max_dim = max(h, w)
            
            if max_dim <= target_res:
                return tensor_bhwc  # No need to resize if already smaller
                
            # Calculate scale factor
            scale = target_res / max_dim
            new_h = int(round(h * scale / 2)) * 2  # Round to nearest even
            new_w = int(round(w * scale / 2)) * 2  # Round to nearest even
            
            # Ensure minimum size
            new_h = max(new_h, 2)
            new_w = max(new_w, 2)
            
            print(f"Resizing from ({h}, {w}) to ({new_h}, {new_w}) for processing (scale: {scale:.3f})")
            
            # Permute to BCHW for interpolation
            tensor_bchw = tensor_bhwc.permute(0, 3, 1, 2)
            resized_bchw = F.interpolate(tensor_bchw, size=(new_h, new_w), mode='bicubic', align_corners=False)
            return resized_bchw.permute(0, 2, 3, 1)  # Back to BHWC
        
        def _resize_auxiliary_map(aux_map: Optional[torch.Tensor], target_res: int) -> Optional[torch.Tensor]:
            """Resize auxiliary map to processing resolution if provided."""
            if aux_map is None:
                return None
            
            # Convert to BHWC format for resizing
            if aux_map.dim() == 2:  # HW -> BHWC
                aux_map = aux_map.unsqueeze(0).unsqueeze(-1)
            elif aux_map.dim() == 3:  # HWC or BHW -> BHWC
                if aux_map.shape[-1] in [1, 3]:  # HWC
                    aux_map = aux_map.unsqueeze(0)
                else:  # BHW
                    aux_map = aux_map.unsqueeze(-1)
            
            return _resize_to_processing_resolution(aux_map, target_res)
        
        # Resize input image and auxiliary maps to processing resolution
        print(f"Original input image dimensions: {input_image.shape}")
        input_image = _resize_to_processing_resolution(input_image, processing_resolution)
        print(f"Processing input image dimensions: {input_image.shape}")
        
        # Resize auxiliary maps
        if SAM_map is not None:
            SAM_map = _resize_auxiliary_map(SAM_map, processing_resolution)
        
        if depth_map is not None:
            depth_map = _resize_auxiliary_map(depth_map, processing_resolution)
            
        if canny_map is not None:
            canny_map = _resize_auxiliary_map(canny_map, processing_resolution)
            
        if hed_map is not None:
            hed_map = _resize_auxiliary_map(hed_map, processing_resolution)
        
        # Update dimensions after resizing
        B, H, W, C = input_image.shape

        # --- Process seed_locations mask ---
        print(f"Processing seed_locations mask...")
        if seed_locations.dim() == 2:  # HW -> BHW
            seed_locations = seed_locations.unsqueeze(0)
        if seed_locations.shape[0] != B:
            print(f"Warning: seed_locations batch size {seed_locations.shape[0]} != input batch size {B}. Repeating mask.")
            seed_locations = seed_locations[0:1, ...].repeat(B, 1, 1)
        if seed_locations.shape[1:3] != (H, W):
            print(f"Warning: Resizing seed_locations from {seed_locations.shape[1:3]} to {(H,W)}")
            seed_locations = F.interpolate(seed_locations.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
        seed_locations = seed_locations.to(device).float()

        # --- Determine Base Mask (Fill Mask) ---
        if fill_mask is not None:
            print(f"Processing fill_mask...")
            if fill_mask.dim() == 2:  # HW -> BHW
                fill_mask = fill_mask.unsqueeze(0)
            if fill_mask.shape[0] != B:
                print(f"Warning: fill_mask batch size {fill_mask.shape[0]} != input batch size {B}. Repeating mask.")
                fill_mask = fill_mask[0:1, ...].repeat(B, 1, 1)
            if fill_mask.shape[1:3] != (H, W):
                print(f"Warning: Resizing fill_mask from {fill_mask.shape[1:3]} to {(H,W)}")
                fill_mask = F.interpolate(fill_mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            base_mask = fill_mask.to(device).float()
        else:
            # Create a default mask of ones the full size of the input image:
            base_mask = torch.ones((B, H, W), dtype=torch.float32, device=device)
            
        # --- Prepare Optional Auxiliary Maps ---
        # Helper to ensure maps are [B, H, W], float, on correct device
        def _prep_aux(aux: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
            if aux is None: return None
            print(f"Preparing aux map: {name}")
            # Ensure 4D BHWC
            if aux.dim() == 2: # HW -> BHWC
                aux = aux.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]
            elif aux.dim() == 3: # HWC -> BHWC or BHW -> BHWC
                if aux.shape[-1] == 1 or aux.shape[-1] == 3: # HWC
                     aux = aux.unsqueeze(0) # [1, H, W, C]
                else: # Assume BHW
                     aux = aux.unsqueeze(-1) # [B, H, W, 1]
            
            # Ensure Batch dim matches input_image
            if aux.shape[0] != B:
                 print(f"Warning: {name} batch size {aux.shape[0]} != input batch size {B}. Repeating map.")
                 aux = aux[0:1,...].repeat(B, 1, 1, 1) # Take first, repeat

            # Ensure H, W match input_image
            if aux.shape[1:3] != (H, W):
                print(f"Warning: Resizing {name} from {aux.shape[1:3]} to {(H,W)}")
                # Permute to BCHW for interpolate
                aux_bchw = aux.permute(0, 3, 1, 2)
                # Handle grayscale or RGB resize
                target_channels = aux_bchw.shape[1]
                aux_resized_bchw = F.interpolate(aux_bchw.float(), size=(H, W), mode='bilinear', align_corners=False)
                # Permute back to BHWC
                aux = aux_resized_bchw.permute(0, 2, 3, 1)

            # Ensure single channel (take luminance if RGB)
            if aux.shape[-1] == 3:
                print(f"Converting {name} to grayscale.")
                aux = 0.299*aux[...,0] + 0.587*aux[...,1] + 0.114*aux[...,2] # [B, H, W]
            elif aux.shape[-1] > 1:
                 print(f"Warning: {name} has {aux.shape[-1]} channels. Taking the first one.")
                 aux = aux[..., 0] # [B, H, W]
            else: # Already single channel
                aux = aux.squeeze(-1) # [B, H, W]
            
            print(f"Processed {name} dimensions: {aux.shape}")
            return aux.to(device).float() # Ensure float and correct device

        # Helper specifically for SAM RGB maps (preserves RGB channels)
        def _prep_sam_rgb(sam_map: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if sam_map is None: 
                return None
            print("Preparing SAM RGB map...")
            
            # Ensure 4D BHWC
            if sam_map.dim() == 2:  # HW -> BHWC
                sam_map = sam_map.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)  # Convert to RGB
            elif sam_map.dim() == 3:  # HWC -> BHWC
                if sam_map.shape[-1] == 1:  # Grayscale -> RGB
                    sam_map = sam_map.expand(-1, -1, 3).unsqueeze(0)
                elif sam_map.shape[-1] == 3:  # RGB -> BHWC
                    sam_map = sam_map.unsqueeze(0)
                else:  # Assume BHW -> BHWC RGB
                    sam_map = sam_map.unsqueeze(-1).expand(-1, -1, -1, 3)
            
            # Ensure batch dimension matches
            if sam_map.shape[0] != B:
                print(f"Warning: SAM batch size {sam_map.shape[0]} != input batch size {B}. Repeating map.")
                sam_map = sam_map[0:1, ...].repeat(B, 1, 1, 1)
            
            # Ensure spatial dimensions match
            if sam_map.shape[1:3] != (H, W):
                print(f"Warning: Resizing SAM map from {sam_map.shape[1:3]} to {(H,W)}")
                sam_bchw = sam_map.permute(0, 3, 1, 2)  # BHWC -> BCHW
                sam_resized_bchw = F.interpolate(sam_bchw.float(), size=(H, W), mode='bilinear', align_corners=False)
                sam_map = sam_resized_bchw.permute(0, 2, 3, 1)  # BCHW -> BHWC
            
            # Ensure 3 channels (RGB)
            if sam_map.shape[-1] == 1:  # Grayscale -> RGB
                sam_map = sam_map.expand(-1, -1, -1, 3)
            elif sam_map.shape[-1] > 3:  # Take first 3 channels
                print(f"Warning: SAM map has {sam_map.shape[-1]} channels. Taking first 3 for RGB.")
                sam_map = sam_map[..., :3]
            
            print(f"Processed SAM RGB dimensions: {sam_map.shape}")
            return sam_map.to(device).float()

        depth_map_bhw = _prep_aux(depth_map, "Depth")
        canny_map_bhw = _prep_aux(canny_map, "Canny")
        hed_map_bhw = _prep_aux(hed_map, "HED")
        # Prepare SAM as RGB BHWC map
        sam_rgb_bhwc = _prep_sam_rgb(SAM_map)

        # --- Configure and Initialize Fill ---
        fill = OrganicFillBatch(
            input_image=input_image, # Pass original BHWC image
            base_mask=base_mask,     # Pass BHW mask
            seed_locations=seed_locations, # Pass BHW seed locations
            depth=depth_map_bhw,     # Pass BHW depth
            canny=canny_map_bhw,     # Pass BHW canny
            hed=hed_map_bhw,         # Pass BHW hed
            sam=sam_rgb_bhwc,        # Pass BHW samantic probability
            config=cfg
        )

        # --- Run Fill Steps ---
        frames: List[torch.Tensor] = [] # Store ALL BHW frames during simulation
        step = 0
        
        print(f"Starting fill loop with max_steps={max_steps}")
        while step < max_steps:
            boundary_pixels, fill_ratio = fill.step()
            frames.append(fill.get_frame())

            # DEBUG: Print progress every 50 steps or when no boundary pixels
            if step % 50 == 0 or boundary_pixels == 0:
                print(f"Step {step}: boundary_pixels={boundary_pixels}, fill_ratio={fill_ratio:.4f}")
                
            if fill.is_complete():
                print(f"Fill completed early at step {step} due to is_complete() = True.")
                print(f"--> Final fill ratio: {fill_ratio:.4f}")
                break
                
            # DEBUG: Early warning if stuck
            if step > 100 and boundary_pixels == 0:
                print(f"WARNING: No boundary pixels at step {step}, but not marked complete!")
                print(f"  Current fill ratio: {fill_ratio:.4f}")

            step += 1

        final_mask_bhw = fill.get_frame() # BHW float mask
        total_time = time.time() - start_time
        print(f"Organic fill finished in {total_time:.2f}s after {step} steps.")
        print(f"Final fill ratio: {fill.fill_ratio().mean().item():.4f}")

        # --- Extract exactly n_frames equally spaced frames ---
        if not frames: # Handle case where no frames were generated (shouldn't happen)
            print("Organic fill warning: No frames were generated.")
            # Just return n_frames of ones:
            selected_frames = [torch.ones((B, H, W), device=device, dtype=torch.float32)] * n_frames
        elif len(frames) < n_frames:
            # Repeat the last frame to reach n_frames:
            selected_frames = frames + [frames[-1]] * (n_frames - len(frames))
        else:
            # Extract exactly n_frames equally spaced frames
            if n_frames == 1:
                selected_frames = [frames[-1]]
            else:
                # Create equally spaced indices from 0 to len(frames)-1
                frame_indices = [int(i * (len(frames) - 1) / (n_frames - 1)) for i in range(n_frames)]
                selected_frames = [frames[i] for i in frame_indices]
            print(f"Extracted {len(selected_frames)} equally spaced frames from {len(frames)} total frames")

        # --- Prepare Outputs ---
        # Final Mask: BHW (ComfyUI MASK format)
        # Frames Preview: Stack frames [N, B, H, W], convert to NBHWC for IMAGE type
        if not selected_frames: # Handle case where no frames were selected (should not happen)
            print("Warning: No frames were selected.")
            frames_preview_nbhwc = torch.zeros((1, B, H, W, 1), device=device, dtype=torch.float32) # Placeholder
            overlayed_fill_preview_nhwc = torch.zeros((1, H, W, 3), device=device, dtype=torch.float32) # Placeholder
        else:
            frames_nbhw = torch.stack(selected_frames) # [N, B, H, W]
            # Convert NBHW (0/1 float) to NBHWC (grayscale float 0-1) for IMAGE output
            frames_preview_nbhwc = frames_nbhw.unsqueeze(-1).expand(-1, -1, -1, -1, 3) # Repeat channel for RGB

            # For simplicity in ComfyUI previews, often only the first batch item is shown.
            # We'll return all batch items, but downstream nodes might only use frames_preview_nbhwc[:, 0, ...]
            # Decision: Return only first batch item frames for preview to avoid huge tensors if batch > 1
            frames_preview_nhwc = frames_preview_nbhwc[:, 0, :, :, :] # [N, H, W, C]
            print(f"Returning final mask (B={B}, H={H}, W={W}) and frames preview (N={frames_preview_nhwc.shape[0]}, H={H}, W={W}, C=3) for batch item 0.")

            # Create overlayed fill preview: overlay mask animation on input image with configurable alpha
            # Get first batch item of input image for overlay
            input_img_hwc = input_image[0] # [H, W, C]
            # Ensure input image is RGB (3 channels)
            if input_img_hwc.shape[-1] == 1: # Grayscale to RGB
                input_img_hwc = input_img_hwc.expand(-1, -1, 3)
            elif input_img_hwc.shape[-1] > 3: # Take first 3 channels
                input_img_hwc = input_img_hwc[..., :3]
            
            # Get frames for first batch item [N, H, W]
            frames_nhw = frames_nbhw[:, 0, :, :] # [N, H, W]
            N = frames_nhw.shape[0]
            
            # Create overlay with alpha blending
            alpha = cfg.overlay_alpha
            overlayed_frames = []
            
            for i in range(N):
                frame_hw = frames_nhw[i] # [H, W] (0/1 float mask)
                
                # Convert mask to RGB: white for filled areas, transparent for unfilled
                # We'll use red color for the fill mask overlay
                mask_rgb_hwc = torch.zeros_like(input_img_hwc) # [H, W, 3]
                mask_rgb_hwc[..., 0] = frame_hw # Red channel = mask
                
                # Alpha blend: result = input * (1 - alpha * mask) + mask_color * (alpha * mask)
                # For areas where mask is 1, blend with configured alpha
                # For areas where mask is 0, keep original image
                mask_alpha_hw = frame_hw * alpha # [H, W] - alpha only where mask is 1
                mask_alpha_hwc = mask_alpha_hw.unsqueeze(-1).expand(-1, -1, 3) # [H, W, 3]
                mask_alpha_hwc = mask_alpha_hwc.to(input_img_hwc.device)
                
                overlayed_hwc = input_img_hwc * (1 - mask_alpha_hwc) + mask_rgb_hwc * mask_alpha_hwc
                overlayed_frames.append(overlayed_hwc)
            
            overlayed_fill_preview_nhwc = torch.stack(overlayed_frames) # [N, H, W, C]

        # --- Scale Outputs Back to Original Resolution ---
        def _scale_back_to_original(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
            """Scale tensor back to original resolution using bicubic interpolation."""
            if tensor.dim() == 3:  # BHW format (masks)
                # Add channel dimension for interpolation
                tensor_bchw = tensor.unsqueeze(1)  # [B, 1, H, W]
                scaled_bchw = F.interpolate(tensor_bchw, size=(target_h, target_w), mode='bicubic', align_corners=False)
                return scaled_bchw.squeeze(1)  # [B, H, W]
            elif tensor.dim() == 4:  # NHWC format (images)
                # Permute to NCHW for interpolation
                tensor_nchw = tensor.permute(0, 3, 1, 2)  # [N, C, H, W]
                scaled_nchw = F.interpolate(tensor_nchw, size=(target_h, target_w), mode='bicubic', align_corners=False)
                return scaled_nchw.permute(0, 2, 3, 1)  # [N, H, W, C]
            else:
                raise ValueError(f"Unsupported tensor dimensions for scaling: {tensor.shape}")
        
        # Only scale if processing resolution was different from original
        if H != orig_H or W != orig_W:
            print(f"Scaling outputs back from ({H}, {W}) to original resolution ({orig_H}, {orig_W})")
            
            # Scale final mask
            final_mask_bhw = _scale_back_to_original(final_mask_bhw, orig_H, orig_W)
            
            # Scale growth probability map
            fill.grow_prob = _scale_back_to_original(fill.grow_prob, orig_H, orig_W)
            
            # Scale frames preview
            if frames_preview_nhwc.numel() > 0:
                frames_preview_nhwc = _scale_back_to_original(frames_preview_nhwc, orig_H, orig_W)
            
            # Scale overlayed fill preview
            if overlayed_fill_preview_nhwc.numel() > 0:
                overlayed_fill_preview_nhwc = _scale_back_to_original(overlayed_fill_preview_nhwc, orig_H, orig_W)
            
            print(f"Final output dimensions: mask={final_mask_bhw.shape}, frames_preview={frames_preview_nhwc.shape}, grow_prob={fill.grow_prob.shape}, overlayed_preview={overlayed_fill_preview_nhwc.shape}")

        # ComfyUI expects MASK as [B, H, W] and IMAGE as [N, H, W, C] or [B, H, W, C]
        # We return final_mask_bhw and frames_preview_nhwc (frames for first batch item)
        return final_mask_bhw, frames_preview_nhwc, fill.grow_prob, overlayed_fill_preview_nhwc


# ────────────────────────────────────────────────────────────────────────────────
#  TEST FUNCTIONALITY (Example - Not run by default in ComfyUI)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend for saving files
    from fill_utils import load_image, save_final_mask, save_frames_as_gif, visualize_final_result

    def run_test_case(test_config: Dict[str, Any], output_dir: str):
        """Loads data, runs the fill node, and saves visualizations for a single test case."""
        test_name = test_config.get("name", "unnamed_test")
        print(f"\n--- Running test case: {test_name} ---")

        # --- 1. Load Input Image ---
        input_path = test_config.get("input")
        # load_image should return HWC tensor
        input_img_hwc = load_image(input_path)
        # Convert to BHWC
        input_img_bhwc = (input_img_hwc.unsqueeze(0).float()).cpu()
        H, W = input_img_hwc.shape[:2]
        target_size = (H,W)
        print(f"Input image loaded ({test_name}): {input_img_bhwc.shape}, Target size: {target_size}")

        # --- 2. Load Optional Auxiliary Maps ---
        depth_path = test_config.get("depth")
        canny_path = test_config.get("canny")
        hed_path = test_config.get("hed")
        sam_path = test_config.get("sam")

        # load_image returns HWC tensor, convert to BHWC
        def _load_aux(path, size):
            if not path or not os.path.exists(path): return None
            img = load_image(path, size)
            return img.unsqueeze(0).cpu() if img is not None else None # BHWC float

        depth_map_bhwc = _load_aux(depth_path, target_size)
        canny_map_bhwc = _load_aux(canny_path, target_size)
        hed_map_bhwc = _load_aux(hed_path, target_size)
        sam_map_bhwc = _load_aux(sam_path, target_size) # This will be thresholded inside execute

        if sam_map_bhwc is not None: print(f"Loaded SAM map: {sam_map_bhwc.shape}")
        if depth_map_bhwc is not None: print(f"Loaded Depth map: {depth_map_bhwc.shape}")
        if canny_map_bhwc is not None: print(f"Loaded Canny map: {canny_map_bhwc.shape}")

        # --- 4. Prepare Node Parameters ---
        # Create a simple seed_locations mask (center point for testing)
        seed_locations_mask = torch.zeros((1, H, W), dtype=torch.float32)
        # Add a small white blob in the center:
        blob_size = 10
        seed_locations_mask[0, H//2-blob_size:H//2+blob_size, W//2-blob_size:W//2+blob_size] = 1.0

        # Get default config for any missing test parameters
        default_config = FillNodeConfig()

        node_params = {
            "input_image": input_img_bhwc, # BHWC float
            "seed_locations": seed_locations_mask, # BHW binary mask for seed placement (float 0-1)
            "fill_mask": None,             # Optional BHW mask defining fillable regions (float 0-1)
            "SAM_map": sam_map_bhwc,       # BHWC float or None
            "depth_map": depth_map_bhwc,   # BHWC float or None
            "canny_map": canny_map_bhwc,   # BHWC float or None
            "hed_map": hed_map_bhwc,       # BHWC float or None
            "max_steps": test_config.get("max_steps", default_config.max_steps),
            "growth_threshold": test_config.get("growth_threshold", default_config.growth_threshold),
            "barrier_jump_power": test_config.get("barrier_jump_power", default_config.barrier_jump_power),
            "seed": test_config.get("seed", default_config.seed),
            "processing_resolution": test_config.get("processing_resolution", default_config.processing_resolution),
            # Advanced FillNodeConfig parameters
            "saturation_window": test_config.get("saturation_window", default_config.saturation_window),
            "saturation_threshold": test_config.get("saturation_threshold", default_config.saturation_threshold),
        }

        # --- 5. Run Organic Fill Node ---
        fill_node = OrganicFillNode()
        # Returns final_mask (BHW float), frames_preview (NHWC float for batch 0), grow_prob (BHW float), overlayed_fill_preview (NHWC float for batch 0)
        final_mask_bhw, frames_preview_nhwc, grow_prob_bhw, overlayed_fill_preview_nhwc = fill_node.execute(**node_params)

        # --- 6. Visualize & Save Outputs ---
        # Utilities expect HW uint8 mask, NHW uint8 frames, HWC uint8 input

        # Get first batch item for mask, convert HW float -> HW byte
        final_mask_hw_byte = (final_mask_bhw[0] * 255).byte().cpu()
        save_final_mask(output_dir, test_name, final_mask_hw_byte)

        # Save growth probability map for first batch item
        grow_prob_hw_byte = (grow_prob_bhw[0] * 255).byte().cpu()
        grow_prob_pil = Image.fromarray(grow_prob_hw_byte.numpy(), mode='L')
        grow_prob_pil.save(os.path.join(output_dir, f"{test_name}_grow_prob.png"))

        # Convert frames preview NHWC float -> NHW byte
        frames_nhw_byte = (frames_preview_nhwc[..., 0] * 255).byte().cpu() # Take first channel
        save_frames_as_gif(frames_nhw_byte, os.path.join(output_dir, f"{test_name}_fill_process.gif"))
        
        # Save overlayed fill preview as GIF
        # Convert NHWC float -> NHWC byte for GIF creation
        overlayed_nhwc_byte = (overlayed_fill_preview_nhwc * 255).byte().cpu()
        # Convert RGB frames to PIL Images and save as GIF
        overlayed_pil_frames = []
        for i in range(overlayed_nhwc_byte.shape[0]):
            frame_hwc_byte = overlayed_nhwc_byte[i].numpy()
            pil_frame = Image.fromarray(frame_hwc_byte, mode='RGB')
            overlayed_pil_frames.append(pil_frame)
        
        if overlayed_pil_frames:
            overlayed_pil_frames[0].save(
                os.path.join(output_dir, f"{test_name}_overlayed_fill.gif"),
                save_all=True,
                append_images=overlayed_pil_frames[1:],
                duration=100,  # 100ms per frame
                loop=0
            )
        
        # Input image HWC uint8, final mask HW byte
        input_img_hwc_byte = (input_img_bhwc[0] * 255).byte().cpu()
        visualize_final_result(
            output_dir=output_dir,
            test_name=test_name,
            input_img=input_img_hwc_byte,
            final_mask=final_mask_hw_byte
        )

        print(f"--- Test {test_name} complete. Results saved to {output_dir}/ ---")
        print(f"    Generated: {test_name}_final_mask.png, {test_name}_grow_prob.png, {test_name}_fill_process.gif, {test_name}_overlayed_fill.gif, {test_name}_visualization.png")

    # --- Define Test Setup ---
    script_dir = os.path.dirname(__file__)
    test_dir = os.path.join(script_dir, "test_assets")
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Define Test Cases ---
    test_cases = [
        {
            "name": "tree",
            "input": os.path.join(test_dir, "tree.jpg"),
            "depth": os.path.join(test_dir, "tree_depth.jpg"),
            "canny": os.path.join(test_dir, "tree_canny.jpg"),
            "hed": os.path.join(test_dir, "tree_hed.jpg"),
            "sam": os.path.join(test_dir, "tree_sam.jpg"),
        },
        {
            "name": "church",
            "input": os.path.join(test_dir, "church.jpg"),
            "depth": os.path.join(test_dir, "church_depth.jpg"),
            "canny": os.path.join(test_dir, "church_canny.jpg"),
            "hed": os.path.join(test_dir, "church_hed.jpg"),
            "sam": os.path.join(test_dir, "church_sam.jpg"),
        },
        {
            "name": "rock",
            "input": os.path.join(test_dir, "rock.jpg"),
            "depth": os.path.join(test_dir, "rock_depth.jpg"),
            "canny": os.path.join(test_dir, "rock_canny.jpg"),
            "hed": os.path.join(test_dir, "rock_hed.jpg"),
            "sam": os.path.join(test_dir, "rock_sam.jpg"),
        },
    ]

    # --- Run Tests ---
    for config in test_cases[:]:
        run_test_case(config, output_dir)

    print("\nAll specified tests completed!")
