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

class StartField(Enum):
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
class FillNodeConfig:
    growth_threshold: float = 0.6
    seed_radius: int = 5
    stability_threshold: int = 20
    active_region_padding: int = 3
    noise_low: float = 0.0
    noise_high: float = 1.2
    saturation_window: int = 15
    saturation_threshold: float = 1e-3
    lab_gradient_scale: float = 5.0 # Scale factor for LAB L* gradient influence
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Growth probability map weights for weighted combination
    weight_lab: float = 0.01      # Weight for LAB gradient grow_prob
    weight_sam: float = 1.0      # Weight for SAM segmentation grow_prob  
    weight_depth: float = 1.5    # Weight for depth gradient grow_prob
    weight_canny: float = 0.05    # Weight for Canny edge grow_prob
    weight_hed: float = 0.5      # Weight for HED edge grow_prob

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

def _image_bhwc_to_lab_l_bhw(img_bhwc: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor [B,H,W,C] (0-1 float) to LAB L* channel [B,H,W]."""
    B, H, W, C = img_bhwc.shape
    l_channel_batch = []
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
        l_channel_hw = lab_hwc[..., 0] # L* channel is the first one
        l_channel_batch.append(torch.from_numpy(l_channel_hw.astype(np.float32) / 100.0)) # L* is 0-100

    return torch.stack(l_channel_batch).to(device) # [B,H,W]

def _sam_rgb_to_color_transition_penalty(sam_rgb_bhwc: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB SAM segmentation map to color transition penalty map.
    Args:
        sam_rgb_bhwc: [B,H,W,C] RGB segmentation map where each flat color represents a segment
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
        threshold_value = 0.25
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
                 depth: Optional[torch.Tensor] = None,  # [B,H,W] float 0‑1
                 canny: Optional[torch.Tensor] = None,  # [B,H,W] float/bool
                 hed: Optional[torch.Tensor] = None,  # [B,H,W] float/bool
                 sam: Optional[torch.Tensor] = None,    # [B,H,W,C] RGB float 0‑1 (segmentation map)
                 start_field: StartField = StartField.RANDOM,
                 config: FillNodeConfig = None):

        self.cfg = config or FillNodeConfig()
        self.device = torch.device(self.cfg.device)

        self.input_image = input_image.to(self.device)
        self.mask = base_mask.to(self.device).float()  # 1 inside, 0 outside
        # Note: self.mask is currently always torch.ones((B, H, W)) as initialized in execute()
        B, H, W = self.mask.shape
        self.B, self.H, self.W = B, H, W

        # Optional channels
        self.depth = depth.to(self.device) if depth is not None else None
        self.canny = canny.to(self.device) if canny is not None else None
        self.hed   = hed.to(self.device)   if hed is not None else None
        self.sam_rgb = sam.to(self.device) if sam is not None else None  # Store RGB SAM for color transitions
        
        # Grids
        self.grid = torch.zeros_like(self.mask)                 # fill state 0/1
        self.activity_counter = torch.zeros_like(self.mask)     # int32
        self.active_mask = torch.ones_like(self.mask).bool()
        self.frame_count = 0
        self.fill_history: List[torch.Tensor] = []

        # Precompute sobel kernels on device
        global _SOBEL_X, _SOBEL_Y
        _SOBEL_X = _SOBEL_X.to(self.device)
        _SOBEL_Y = _SOBEL_Y.to(self.device)

        print(f"Preparing growth probability fields on {self.device}...")
        self._prepare_grow_prob_fields()
        self._place_seeds(start_field)

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

        # ── LAB L* Gradient Growth Probability (always available) ──
        print("Computing LAB L* gradient growth probability...")
        l_channel = _image_bhwc_to_lab_l_bhw(self.input_image)  # [B,H,W]
        l_grad_mag = _sobel_grad(l_channel)
        lab_grow_prob = torch.exp(-l_grad_mag * self.cfg.lab_gradient_scale)
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

        if save_maps:
            comfy_tensor_to_pil(self.grow_prob.unsqueeze(-1)).save('grow_prob.png')

    # ────────────────────────────────────────────────────────────────────────
    #  SEED PLACEMENT
    # ────────────────────────────────────────────────────────────────────────
    def _place_seeds(self, start_field: StartField):
        B,H,W = self.B,self.H,self.W

        print(f"Placing seeds using {start_field.value} strategy within the mask...")
        for b in range(B):
            # Determine seed coordinates (y, x)
            yx = None

            # Standard region selection
            if yx is None:
                h2, w2 = H // 2, W // 2
                thirds_h, thirds_w = H // 3, W // 3
                sel = torch.zeros_like(self.mask[b], dtype=torch.bool) # Selection region

                if start_field == StartField.RANDOM:
                    sel = self.mask[b] # Select from anywhere inside mask
                elif start_field == StartField.TOP_LEFT:      sel[:h2, :w2] = True
                elif start_field == StartField.TOP_RIGHT:     sel[:h2, w2:] = True
                elif start_field == StartField.BOTTOM_LEFT:   sel[h2:, :w2] = True
                elif start_field == StartField.BOTTOM_RIGHT:  sel[h2:, w2:] = True
                elif start_field == StartField.CENTER:        sel[thirds_h:2*thirds_h, thirds_w:2*thirds_w] = True
                elif start_field == StartField.TOP_CENTER:    sel[:thirds_h, thirds_w:2*thirds_w] = True
                elif start_field == StartField.BOTTOM_CENTER: sel[2*thirds_h:, thirds_w:2*thirds_w] = True
                elif start_field == StartField.LEFT_CENTER:   sel[thirds_h:2*thirds_h, :thirds_w] = True
                elif start_field == StartField.RIGHT_CENTER:  sel[thirds_h:2*thirds_h, 2*thirds_w:] = True

                # Find valid points intersection of mask and selected region
                valid_candidates = torch.nonzero(sel, as_tuple=False) # Simplified: m is True

                if valid_candidates.numel() == 0: # If region+mask is empty, fallback to any point in mask
                    valid_candidates = torch.nonzero(torch.ones_like(sel), as_tuple=False) # Simplified: m is True

                if valid_candidates.numel() > 0:
                    # Randomly pick one from the candidates
                    yx = valid_candidates[torch.randint(len(valid_candidates), (1,))]
                else: # Should not happen if mask `m` was not empty initially
                    print(f"Error: No valid pixels found for seeding in batch {b} even after fallback. Using center.")
                    yx = torch.tensor([[H//2, W//2]], device=self.device, dtype=torch.long)

            # Extract y, x coordinates
            y, x = yx[0, 0], yx[0, 1]

            # Apply circular seed around (y, x), respecting the mask `m`
            yy, xx = torch.meshgrid(torch.arange(H,device=self.device), torch.arange(W,device=self.device), indexing='ij')
            dist_sq = (yy - y)**2 + (xx - x)**2
            circle = dist_sq <= self.cfg.seed_radius**2 # The actual seed shape
            self.grid[b][circle] = 1.0 # Simplified: m is True

        print("Seed placement complete.")


    # ────────────────────────────────────────────────────────────────────────
    #  CORE STEP
    # ────────────────────────────────────────────────────────────────────────
    def _update_activity(self, new_growth: torch.Tensor):
        self.activity_counter[new_growth] = 0
        self.activity_counter[~new_growth] += 1
        # Pixels that have been stable (not grown) for a while
        stable = self.activity_counter >= self.cfg.stability_threshold

        # Active region is defined around pixels that are *not* stable
        active_area = ~stable

        # Dilate the active area to include neighbours
        padding = self.cfg.active_region_padding
        # Use max_pool2d for efficient dilation on GPU
        dilated_active = F.max_pool2d(active_area.float().unsqueeze(1),
                                     kernel_size=2*padding+1,
                                     stride=1,
                                     padding=padding).squeeze(1) > 0

        # Update active mask: must be dilated active area, not yet filled, and within the base mask
        self.active_mask = dilated_active & (self.grid < 1)


    def step(self):
        self.frame_count += 1
        B,H,W = self.B,self.H,self.W

        # Find neighbours of currently filled cells
        padded_grid = F.pad(self.grid.unsqueeze(1), (1,1,1,1), mode='constant', value=0) # Pad [B,H,W] -> [B,1,H+2,W+2]
        neighbours = F.max_pool2d(padded_grid, kernel_size=3, stride=1, padding=0) # Neighbours include self
        # Identify boundary cells: not filled, but have a filled neighbour, and are within the mask
        has_filled_neighbour = (neighbours.squeeze(1) > 0) & (self.grid == 0)

        P = self.grow_prob

        # --- Pulsating threshold ---
        thr = self.cfg.growth_threshold

        # --- Determine potential growth cells ---
        # Must be:
        # 1. Not already filled (grid == 0)
        # 2. Inside the main mask (mask > 0)
        # 3. Have a filled neighbour (has_filled_neighbour)
        # 4. Be within the active region (active_mask)
        # 5. Have growth probability P > threshold (thr)
        potential_growth = (self.grid == 0) & (self.mask > 0) & has_filled_neighbour & self.active_mask & (P > thr)

        # --- Stochastic Growth ---
        # Add noise only where potential growth is possible
        noise = torch.rand_like(self.grid, device=self.device) * (self.cfg.noise_high - self.cfg.noise_low) + self.cfg.noise_low
        # New growth happens where potential exists AND noise < probability
        new_growth = potential_growth & (noise < P)

        # Update grid and activity
        self.grid[new_growth] = 1.0
        self._update_activity(new_growth)

        # Calculate fill ratio and track history
        fill_ratio = self.fill_ratio()
        self.fill_history.append(fill_ratio.clone()) # Store fill ratio per batch item

        # Return active pixel count for progress reporting
        active_count = torch.sum(self.active_mask).item() # Total active pixels across batch
        # Return average fill ratio across batch for simpler reporting
        avg_fill_ratio = fill_ratio.mean().item()
        return active_count, avg_fill_ratio

    # ────────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────────────────────────
    def fill_ratio(self):
        """Calculate fill ratio per batch item (simplified for full mask)."""
        mask_area = torch.full((self.B,), self.H * self.W, device=self.device, dtype=torch.float32)
        filled_area = self.grid.sum(dim=(1,2)).float()
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

        # Check 3: No more active pixels available to fill within the mask (per batch item)
        # Active mask is [B, H, W], sum over H, W -> [B]
        # Simplified: self.mask > 0 is always true
        no_more_active = (self.active_mask & (self.grid == 0)).sum(dim=(1,2)) == 0

        # Complete if *both* saturated AND no more active pixels for that batch item
        # Use logical AND element-wise for the batch
        is_done = saturated & no_more_active

        # The whole process is complete if *all* batch items are done
        return torch.all(is_done)

    def get_frame(self):
        return self.grid.clone()  # [B,H,W] float 0/1

# ────────────────────────────────────────────────────────────────────────────────
#  COMFYUI NODE DEFINITION
# ────────────────────────────────────────────────────────────────────────────────

class OrganicFillNode:
    """ComfyUI node wrapping the confidence‑aware Organic Fill algorithm."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {}),  # BHWC RGB or Gray (float 0-1)
            },
            "optional": {
                "SAM_map": ("IMAGE", {}),      # Optional: BHWC/HWC Probability Map (float 0-1)
                "depth_map": ("IMAGE", {}),    # Optional: BHWC/HWC Depth Map (float 0-1)
                "canny_map": ("IMAGE", {}),    # Optional: BHWC/HWC Canny Edges (float 0-1)
                "hed_map": ("IMAGE", {}),      # Optional: BHWC/HWC HED Edges (float 0-1)
                "start_field": ( [sf.value for sf in StartField], {"default": StartField.CENTER.value} ),
                "max_steps": ("INT", {"default":2000, "min":1, "max":10000}),
                "growth_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed_radius": ("INT", {"default": 5, "min": 1, "max": 50}),
                "stability_threshold": ("INT", {"default": 20, "min": 1, "max": 100}),
                "noise_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "noise_high": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "lab_gradient_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1}), # Only used if no other maps
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
                SAM_map: Optional[torch.Tensor] = None, # BHWC/HWC
                depth_map: Optional[torch.Tensor] = None, # BHWC/HWC
                canny_map: Optional[torch.Tensor] = None, # BHWC/HWC
                hed_map: Optional[torch.Tensor] = None, # BHWC/HWC
                start_field: str = StartField.CENTER.value,
                max_steps: int = 2000,
                growth_threshold: float = 0.6,
                seed_radius: int = 5,
                stability_threshold: int = 20,
                noise_low: float = 0.0,
                noise_high: float = 1.2,
                lab_gradient_scale: float = 5.0,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # Return mask BHW, frames NBHWC, grow_prob BHW, overlayed_fill_preview NHWC

        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"OrganicFill starting on {device} with max {max_steps} steps")
        print(f"Input image dimensions: {input_image.shape}")

        # --- Input Validation and Preparation ---
        if input_image.dim() != 4:
            raise ValueError(f"Expected input_image with shape [B, H, W, C], got {input_image.shape}")
        B, H, W, C = input_image.shape

        # --- Determine Base Mask ---
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
        cfg = FillNodeConfig(
            growth_threshold=growth_threshold,
            seed_radius=seed_radius,
            stability_threshold=stability_threshold,
            noise_low=noise_low,
            noise_high=noise_high,
            lab_gradient_scale=lab_gradient_scale,
            device=device
        )

        fill = OrganicFillBatch(
            input_image=input_image, # Pass original BHWC image
            base_mask=base_mask,     # Pass BHW mask
            depth=depth_map_bhw,     # Pass BHW depth
            canny=canny_map_bhw,     # Pass BHW canny
            hed=hed_map_bhw,         # Pass BHW hed
            sam=sam_rgb_bhwc,        # Pass BHW samantic probability
            start_field=StartField(start_field),
            config=cfg
        )

        # --- Run Fill Steps ---
        frames: List[torch.Tensor] = [] # Store BHW frames
        step = 0
        print("Starting organic fill process...")
        # More frequent reporting initially, then less often
        report_steps = set([0, 1, 2, 3, 4, 5, 10, 20, 50] + list(range(100, max_steps + 1, 100)))
        save_frame_steps = set([0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 50] + list(range(100, max_steps + 1, 50)))

        while step < max_steps:
            active_pixels, fill_ratio = fill.step()

            # Store frame if it's a designated step or the last step
            is_last_step = step == max_steps - 1
            if step in save_frame_steps or is_last_step:
                 frames.append(fill.get_frame().clone()) # Clone BHW tensor

            # Print progress updates at designated steps or if complete
            is_complete = fill.is_complete()
            if step in report_steps or is_last_step or is_complete:
                elapsed = time.time() - start_time
                print(f"Step {step}/{max_steps} | Fill ratio: {fill_ratio:.4f} | Active px: {active_pixels} | Elapsed: {elapsed:.2f}s")

            if is_complete:
                print(f"Fill completed early at step {step} due to saturation or no active pixels.")
                # Ensure the very last frame is captured if completed early
                if not frames or not torch.all(frames[-1] == fill.get_frame()):
                    frames.append(fill.get_frame().clone())
                break

            step += 1

        # If loop finished by max_steps, ensure last frame is added if needed
        if step == max_steps and (not frames or not torch.all(frames[-1] == fill.get_frame())):
             frames.append(fill.get_frame().clone())

        final_mask_bhw = fill.get_frame() # BHW float mask
        total_time = time.time() - start_time
        print(f"Organic fill finished in {total_time:.2f}s after {step} steps.")

        # --- Prepare Outputs ---
        # Final Mask: BHW (ComfyUI MASK format)
        # Frames Preview: Stack frames [N, B, H, W], convert to NBHWC for IMAGE type
        if not frames: # Handle case where no frames were saved (e.g., max_steps=0)
            print("Warning: No frames were generated.")
            frames_preview_nbhwc = torch.zeros((1, B, H, W, 1), device=device, dtype=torch.float32) # Placeholder
            overlayed_fill_preview_nhwc = torch.zeros((1, H, W, 3), device=device, dtype=torch.float32) # Placeholder
        else:
            frames_nbhw = torch.stack(frames) # [N, B, H, W]
            # Convert NBHW (0/1 float) to NBHWC (grayscale float 0-1) for IMAGE output
            frames_preview_nbhwc = frames_nbhw.unsqueeze(-1).expand(-1, -1, -1, -1, 3) # Repeat channel for RGB

            # For simplicity in ComfyUI previews, often only the first batch item is shown.
            # We'll return all batch items, but downstream nodes might only use frames_preview_nbhwc[:, 0, ...]
            # Let's just return the first batch item's frames for preview? Or all?
            # Let's return all for flexibility, but maybe just first batch item is better for preview node?
            # Decision: Return only first batch item frames for preview to avoid huge tensors if batch > 1
            frames_preview_nhwc = frames_preview_nbhwc[:, 0, :, :, :] # [N, H, W, C]
            print(f"Returning final mask (B={B}, H={H}, W={W}) and frames preview (N={frames_preview_nhwc.shape[0]}, H={H}, W={W}, C=3) for batch item 0.")

            # Create overlayed fill preview: overlay mask animation on input image with alpha=0.5
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
            alpha = 0.4
            overlayed_frames = []
            
            for i in range(N):
                frame_hw = frames_nhw[i] # [H, W] (0/1 float mask)
                
                # Convert mask to RGB: white for filled areas, transparent for unfilled
                # We'll use red color for the fill mask overlay
                mask_rgb_hwc = torch.zeros_like(input_img_hwc) # [H, W, 3]
                mask_rgb_hwc[..., 0] = frame_hw # Red channel = mask
                
                # Alpha blend: result = input * (1 - alpha * mask) + mask_color * (alpha * mask)
                # For areas where mask is 1, blend with alpha=0.5
                # For areas where mask is 0, keep original image
                mask_alpha_hw = frame_hw * alpha # [H, W] - alpha only where mask is 1
                mask_alpha_hwc = mask_alpha_hw.unsqueeze(-1).expand(-1, -1, 3) # [H, W, 3]
                mask_alpha_hwc = mask_alpha_hwc.to(input_img_hwc.device)
                
                overlayed_hwc = input_img_hwc * (1 - mask_alpha_hwc) + mask_rgb_hwc * mask_alpha_hwc
                overlayed_frames.append(overlayed_hwc)
            
            overlayed_fill_preview_nhwc = torch.stack(overlayed_frames) # [N, H, W, C]

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
        node_params = {
            "input_image": input_img_bhwc, # BHWC float
            "SAM_map": sam_map_bhwc,       # BHWC float or None
            "depth_map": depth_map_bhwc,   # BHWC float or None
            "canny_map": canny_map_bhwc,   # BHWC float or None
            "hed_map": hed_map_bhwc,       # BHWC float or None
            "max_steps": test_config.get("max_steps", 1000),
            "start_field": test_config.get("start_field", StartField.CENTER.value),
            "growth_threshold": test_config.get("growth_threshold", 0.6),
            "seed_radius": test_config.get("seed_radius", 5),
            "stability_threshold": test_config.get("stability_threshold", 20),
            "noise_low": test_config.get("noise_low", 0.0),
            "noise_high": test_config.get("noise_high", 1.2),
            "lab_gradient_scale": test_config.get("lab_gradient_scale", 5.0),
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
            "start_field": StartField.BOTTOM_CENTER.value
        },
        {
            "name": "church",
            "input": os.path.join(test_dir, "church.jpg"),
            "depth": os.path.join(test_dir, "church_depth.jpg"),
            "canny": os.path.join(test_dir, "church_canny.jpg"),
            "hed": os.path.join(test_dir, "church_hed.jpg"),
            "sam": os.path.join(test_dir, "church_sam.jpg"),
            "start_field": StartField.BOTTOM_CENTER.value
        },
        {
            "name": "rock",
            "input": os.path.join(test_dir, "rock.jpg"),
            "depth": os.path.join(test_dir, "rock_depth.jpg"),
            "canny": os.path.join(test_dir, "rock_canny.jpg"),
            "hed": os.path.join(test_dir, "rock_hed.jpg"),
            "sam": os.path.join(test_dir, "rock_sam.jpg"),
            "start_field": StartField.BOTTOM_CENTER.value
        },
    ]

    # --- Run Tests ---
    for config in test_cases[:]:
        run_test_case(config, output_dir)

    print("\nAll specified tests completed!")
