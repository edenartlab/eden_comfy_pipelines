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
from torchvision.transforms import ToTensor, ToPILImage, Resize

# Import utilities from the new file
from fill_utils import *

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
    CLOSEST_POINT = "closest_point"  # new option – uses depth or COM

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
    pulsate_strength: float = 0.0  # added – sinusoidal modulation strength
    directional_flow: bool = False  # added – bias by depth gradient
    branch_awareness: float = 0.0  # added – favour skeleton proximity
    lab_gradient_scale: float = 5.0 # Scale factor for LAB L* gradient influence
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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

def _normalise(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise tensor to [0, 1] range per item in batch."""
    if t is None: return None
    if t.dim() < 3: # Handle cases like [H,W] or single values
        mn, mx = t.min(), t.max()
        if mx - mn < eps: return torch.zeros_like(t)
        return (t - mn) / (mx - mn + eps)
    else: # Handle batched tensors [B, H, W] or [B, C, H, W], etc.
        # Normalise each element in the batch independently
        t_norm = torch.zeros_like(t, dtype=torch.float32)
        for i in range(t.shape[0]):
            mn, mx = t[i].min(), t[i].max()
            if mx - mn < eps:
                t_norm[i] = torch.zeros_like(t[i])
            else:
                t_norm[i] = (t[i] - mn) / (mx - mn + eps)
        return t_norm

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


# ────────────────────────────────────────────────────────────────────────────────
#  CORE ORGANIC‑FILL IMPLEMENTATION (BATCH‑AWARE)
# ────────────────────────────────────────────────────────────────────────────────

class OrganicFillBatch:
    """Performs confidence‑aware organic fill on a batch of images."""
    def __init__(self,
                 input_image: torch.Tensor, # [B,H,W,C] float 0-1 RGB/Gray
                 base_mask: torch.Tensor,   # [B,H,W] float 0/1
                 depth: Optional[torch.Tensor] = None,  # [B,H,W] float 0‑1
                 canny: Optional[torch.Tensor] = None,  # [B,H,W] float/bool
                 sem: Optional[torch.Tensor] = None,    # [B,H,W] float 0‑1
                 start_field: StartField = StartField.RANDOM,
                 config: FillNodeConfig = None):

        self.cfg = config or FillNodeConfig()
        self.device = torch.device(self.cfg.device)

        self.input_image = input_image.to(self.device)
        self.mask = base_mask.to(self.device).float()  # 1 inside, 0 outside
        B, H, W = self.mask.shape
        self.B, self.H, self.W = B, H, W

        # Optional channels
        self.depth = depth.to(self.device) if depth is not None else None
        self.canny = canny.to(self.device) if canny is not None else None
        self.sem   = sem.to(self.device)   if sem   is not None else None

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

        print(f"Preparing confidence fields on {self.device}...")
        self._prepare_confidence_fields()
        self._place_seeds(start_field)

    # ────────────────────────────────────────────────────────────────────────
    #  CONFIDENCE FIELD
    # ────────────────────────────────────────────────────────────────────────
    def _prepare_confidence_fields(self):
        """Compute confidence contributions based on available maps."""
        B, H, W = self.B, self.H, self.W
        device = self.device

        # --- Base Confidence: LAB L* gradient if no other maps ---
        # Use LAB L* gradient magnitude as the primary driver if no other guidance is given
        # Favours filling areas of low contrast first.
        has_sem = self.sem is not None
        has_depth = self.depth is not None
        has_canny = self.canny is not None
        use_lab_base = not (has_sem or has_depth or has_canny)

        if use_lab_base:
            print("No specific maps provided, using LAB L* gradient as base confidence.")
            l_channel = _image_bhwc_to_lab_l_bhw(self.input_image) # [B,H,W]
            l_grad_mag = _sobel_grad(l_channel)
            # Inverse relationship: high gradient -> low confidence
            base_confidence = torch.exp(-l_grad_mag * self.cfg.lab_gradient_scale)
            base_confidence = _normalise(base_confidence) # Ensure [0,1]
        else:
            # Start with neutral confidence if using specific maps
            base_confidence = torch.ones((B, H, W), device=device)

        # --- Semantic Map Contribution (if available) ---
        P_sem = torch.ones((B, H, W), device=device)
        if has_sem:
            print("Using Semantic map contribution.")
            P_sem = _normalise(self.sem) # Assume higher value means more likely to be filled

        # --- Signed Distance Field Penalty (always calculated based on base_mask) ---
        # Penalises growing close to the boundary defined by the mask
        print("Computing distance transform (SDF)...")
        sdf_out = []
        mask_cpu = (self.mask > 0).cpu().numpy()
        for b in range(B):
            # If mask is all ones, distance is 0 everywhere, doesn't hurt
            dist_out = cv2.distanceTransform(np.uint8(1 - mask_cpu[b])*255, cv2.DIST_L2, 5)
            sdf_out.append(torch.from_numpy(dist_out))
        SDF_OUT = torch.stack(sdf_out).to(device)
        SDF_OUT = _normalise(SDF_OUT)
        # Increase penalty for being near the border (outside mask)
        sdf_penalty = 1.0 - torch.tanh(SDF_OUT * 3) # High SDF_OUT means far from border -> penalty close to 1

        # --- Depth Gradient Contribution (if available) ---
        # Penalises growing across sharp depth changes
        depth_term = torch.ones((B, H, W), device=device)
        if has_depth:
            print("Using Depth map contribution.")
            d_norm = _normalise(self.depth)
            grad_mag = _sobel_grad(d_norm)  # [B,H,W]
            # Exponential decay: High gradient -> low term value
            depth_term = torch.exp(-grad_mag * 10) # Factor 10 enhances effect

        # --- Canny Edge Penalty (if available) ---
        # Penalises growing near strong edges
        edge_penalty = torch.ones((B, H, W), device=device)
        if has_canny:
            print("Using Canny map contribution.")
            # Assume Canny map is 0-1, where 1 is strong edge
            c_norm = _normalise(self.canny.float())
            edge_penalty = 1.0 - c_norm # Prefer pixels *away* from strong edges

        # --- Combine Confidence Components ---
        # Start with the base (either LAB-based or ones)
        confidence = base_confidence

        # Multiply positive/neutral factors
        if has_sem:
            confidence = confidence * P_sem # Favor regions indicated by SEM map
        if has_depth:
            confidence = confidence * depth_term # Penalize high depth gradients
        confidence = confidence * sdf_penalty # Penalize proximity to mask border (if mask isn't full)
        confidence = confidence * edge_penalty # Penalize proximity to Canny edges

        # Final clamp
        self.confidence = torch.clamp(confidence, 0.0, 1.0)

        # --- Branch Awareness (if enabled and mask is not full) ---
        mask_is_full = self.mask.all() # Check if the mask covers the entire image
        if self.cfg.branch_awareness > 0.0 and not mask_is_full:
            print("Computing branch awareness map...")
            try:
                from skimage.morphology import skeletonize
            except ImportError:
                print("Warning: skimage not found, branch awareness disabled")
                self.branch_weight = torch.ones((B,H,W), device=device)
                return

            bw = []
            mask_cpu_bool = (self.mask > 0).cpu().numpy()
            for b in range(B):
                sk = skeletonize(mask_cpu_bool[b]).astype(np.uint8)
                if np.sum(sk) == 0: # Handle case where skeletonization yields nothing
                     print(f"Warning: Skeletonization of mask for batch {b} resulted in empty skeleton. Branch weight set to 1.")
                     dist_norm = np.zeros((H,W), dtype=np.float32) # Set distance to 0 -> weight to 1
                else:
                    # Calculate distance to the non-zero skeleton pixels
                    dist_to_skel = cv2.distanceTransform(1-sk, cv2.DIST_L2, 3) # Distance to nearest 1 (skeleton)
                    # Normalize distance robustly
                    max_dist = dist_to_skel.max()
                    dist_norm = dist_to_skel / (max_dist + 1e-8) if max_dist > 0 else dist_to_skel
                # Weight is inversely proportional to distance (1 near skeleton, 0 far)
                bw.append(1.0 - torch.from_numpy(dist_norm))
            self.branch_weight = torch.stack(bw).to(device).float()
            print("Branch awareness map computed.")
        else:
            if self.cfg.branch_awareness > 0.0 and mask_is_full:
                print("Branch awareness skipped: base_mask covers the entire image.")
            self.branch_weight = torch.ones((B,H,W), device=device)


    # ────────────────────────────────────────────────────────────────────────
    #  SEED PLACEMENT
    # ────────────────────────────────────────────────────────────────────────
    def _place_seeds(self, start_field: StartField):
        B,H,W = self.B,self.H,self.W
        # Valid pixels for seeding are within the provided base_mask
        valid = self.mask > 0

        print(f"Placing seeds using {start_field.value} strategy within the mask...")
        for b in range(B):
            m = valid[b] # Mask for the current batch item
            if not m.any(): # Check if the mask is completely empty
                print(f"Warning: Mask for batch {b} is empty. Cannot place seed. Organic fill may not work.")
                continue # Skip seed placement for this item

            # Determine seed coordinates (y, x)
            yx = None
            if start_field == StartField.CLOSEST_POINT and self.depth is not None:
                depth_b = self.depth[b]
                # Find min depth only within the valid mask region
                depth_inside = torch.where(m, depth_b, torch.full_like(depth_b, float('inf')))
                min_depth_val = depth_inside.min()

                if torch.isfinite(min_depth_val):
                    yx_candidates = torch.nonzero(depth_inside == min_depth_val, as_tuple=False)
                    if yx_candidates.numel() > 0:
                        # Randomly pick one if multiple minima exist
                        yx = yx_candidates[torch.randint(len(yx_candidates), (1,))]
                # Fallback if no finite min depth found within the mask
                if yx is None:
                    print(f"Warning: Could not find finite minimum depth within mask for batch {b}. Falling back.")

            # Fallback or standard region selection
            if yx is None:
                h2, w2 = H // 2, W // 2
                thirds_h, thirds_w = H // 3, W // 3
                sel = torch.zeros_like(m, dtype=torch.bool) # Selection region

                if start_field == StartField.RANDOM or start_field == StartField.CLOSEST_POINT: # CLOSEST_POINT falls back to RANDOM if depth fails
                    sel = m # Select from anywhere inside mask
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
                valid_candidates = torch.nonzero(m & sel, as_tuple=False)

                if valid_candidates.numel() == 0: # If region+mask is empty, fallback to any point in mask
                    valid_candidates = torch.nonzero(m, as_tuple=False)

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
            circle = dist_sq <= self.cfg.seed_radius**2
            self.grid[b][circle & m] = 1.0 # Seed only within the mask

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
        self.active_mask = dilated_active & (self.grid < 1) & (self.mask > 0)


    def step(self):
        self.frame_count += 1
        B,H,W = self.B,self.H,self.W

        # Find neighbours of currently filled cells
        padded_grid = F.pad(self.grid.unsqueeze(1), (1,1,1,1), mode='constant', value=0) # Pad [B,H,W] -> [B,1,H+2,W+2]
        neighbours = F.max_pool2d(padded_grid, kernel_size=3, stride=1, padding=0) # Neighbours include self
        # Identify boundary cells: not filled, but have a filled neighbour, and are within the mask
        has_filled_neighbour = (neighbours.squeeze(1) > 0) & (self.grid == 0)

        # Calculate base growth probability P = confidence * branch_weight
        # Branch weight is 1 if awareness is off or mask is full
        P = self.confidence * self.branch_weight

        # --- Directional flow bias via depth gradient (if enabled and depth available) ---
        if self.cfg.directional_flow and self.depth is not None:
            # Calculate depth gradient magnitude
            d_norm = _normalise(self.depth)
            grad_mag = _sobel_grad(d_norm)
            grad_mag_norm = _normalise(grad_mag) # Normalize gradient magnitude 0-1

            # Bias probability: Increase probability in direction of *lower* depth gradient
            # We use (1 - grad_mag_norm) - areas with low gradient get higher bias
            # Additive bias factor - adjust strength as needed
            depth_bias_strength = 0.5 # Example strength factor
            flow_bias = (1.0 - grad_mag_norm) * depth_bias_strength
            P = P * (1.0 + flow_bias) # Modulate base probability
            P = torch.clamp(P, 0, 1) # Ensure probability stays within [0,1]

        # --- Pulsating threshold ---
        thr = self.cfg.growth_threshold
        if self.cfg.pulsate_strength > 0:
            thr += self.cfg.pulsate_strength * math.sin(2 * math.pi * self.frame_count / 60) # 60 steps per cycle


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
        """Calculate fill ratio per batch item."""
        mask_area = self.mask.sum(dim=(1,2))
        filled_area = (self.grid * self.mask).sum(dim=(1,2))
        # Avoid division by zero if mask area is zero for some batch items
        ratio = torch.where(mask_area > 0, filled_area / mask_area, torch.zeros_like(mask_area))
        return ratio # [B]

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
        no_more_active = (self.active_mask & (self.grid == 0) & (self.mask > 0)).sum(dim=(1,2)) == 0

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
                 # If SAM_map is provided, it acts as the base_mask after thresholding.
                 # If not provided, base_mask becomes the full image area.
                "SAM_map": ("IMAGE", {}),      # Optional: BHWC/HWC Probability Map (float 0-1)
                "depth_map": ("IMAGE", {}),    # Optional: BHWC/HWC Depth Map (float 0-1)
                "canny_map": ("IMAGE", {}),    # Optional: BHWC/HWC Canny Edges (float 0-1)
                "start_field": ( [sf.value for sf in StartField], {"default": StartField.CENTER.value} ),
                "pulsate_strength": ("FLOAT", {"default": 0.0, "min":0.0, "max":1.0, "step":0.01}),
                "directional_flow": ("BOOLEAN", {"default": False}), # Bias growth based on depth gradient
                "branch_awareness": ("FLOAT", {"default": 0.0, "min":0.0, "max":1.0, "step":0.01}), # Prefer skeleton
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
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("final_mask", "frames_preview") # frames need conversion for IMAGE type

    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/Experimental"

    # ────────────────────────────────────────────────────────────────────────
    def execute(self,
                input_image: torch.Tensor, # BHWC, float 0-1
                SAM_map: Optional[torch.Tensor] = None, # BHWC/HWC
                depth_map: Optional[torch.Tensor] = None, # BHWC/HWC
                canny_map: Optional[torch.Tensor] = None, # BHWC/HWC
                start_field: str = StartField.CENTER.value,
                pulsate_strength: float = 0.0,
                directional_flow: bool = False,
                branch_awareness: float = 0.0,
                max_steps: int = 2000,
                growth_threshold: float = 0.6,
                seed_radius: int = 5,
                stability_threshold: int = 20,
                noise_low: float = 0.0,
                noise_high: float = 1.2,
                lab_gradient_scale: float = 5.0,
                ) -> Tuple[torch.Tensor, torch.Tensor]: # Return mask BHW, frames NBHWC

        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"OrganicFill starting on {device} with max {max_steps} steps")
        print(f"Input image dimensions: {input_image.shape}")

        # --- Input Validation and Preparation ---
        if input_image.dim() != 4:
            raise ValueError(f"Expected input_image with shape [B, H, W, C], got {input_image.shape}")
        B, H, W, C = input_image.shape

        # --- Determine Base Mask ---
        if SAM_map is not None:
            print("SAM map provided, using it to create base mask.")
            # Ensure SAM_map is BHWC, take first channel, threshold
            if SAM_map.dim() == 3: # HWC -> BHWC
                SAM_map = SAM_map.unsqueeze(0)
            if SAM_map.shape[0] != B: # Repeat if batch size mismatch (e.g., single mask for batch)
                 print(f"Warning: SAM map batch size {SAM_map.shape[0]} != input batch size {B}. Repeating SAM map.")
                 SAM_map = SAM_map.repeat(B, 1, 1, 1)
            if SAM_map.shape[1:3] != (H, W):
                print(f"Warning: Resizing SAM map from {SAM_map.shape[1:3]} to {(H,W)}")
                SAM_map = F.interpolate(SAM_map.permute(0,3,1,2), size=(H,W), mode='bilinear', align_corners=False).permute(0,2,3,1)

            sam_prob = SAM_map[..., 0] # Use first channel as probability [B, H, W]
            base_mask = (sam_prob > 0.5).float() # Threshold to get binary mask [B, H, W]
            print(f"Base mask created from SAM map. Shape: {base_mask.shape}")
        else:
            print("No SAM map provided. Base mask covers the entire image.")
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

        depth_map_bhw = _prep_aux(depth_map, "Depth")
        canny_map_bhw = _prep_aux(canny_map, "Canny")
        # SAM map was already handled for base_mask, but we need the probability map for confidence
        # Re-use the prepared SAM_map if it existed, otherwise it's None
        sam_prob_bhw = _prep_aux(SAM_map, "SAM_Prob") if SAM_map is not None else None

        # --- Configure and Initialize Fill ---
        cfg = FillNodeConfig(
            growth_threshold=growth_threshold,
            seed_radius=seed_radius,
            stability_threshold=stability_threshold,
            noise_low=noise_low,
            noise_high=noise_high,
            pulsate_strength=pulsate_strength,
            directional_flow=directional_flow,
            branch_awareness=branch_awareness,
            lab_gradient_scale=lab_gradient_scale,
            device=device
        )

        fill = OrganicFillBatch(
            input_image=input_image, # Pass original BHWC image
            base_mask=base_mask,     # Pass BHW mask (full or from SAM)
            depth=depth_map_bhw,     # Pass BHW depth
            canny=canny_map_bhw,     # Pass BHW canny
            sem=sam_prob_bhw,        # Pass BHW semantic probability
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


        # ComfyUI expects MASK as [B, H, W] and IMAGE as [N, H, W, C] or [B, H, W, C]
        # We return final_mask_bhw and frames_preview_nhwc (frames for first batch item)
        return final_mask_bhw, frames_preview_nhwc


# ────────────────────────────────────────────────────────────────────────────────
#  TEST FUNCTIONALITY (Example - Not run by default in ComfyUI)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend for saving files
    # Assume fill_utils provides these functions if running standalone
    from fill_utils import load_image, save_final_mask, save_frames_as_gif, visualize_inputs, visualize_frames_grid, visualize_final_result

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
        sam_path = test_config.get("SAM")

        # load_image returns HWC uint8, convert to float 0-1
        def _load_aux(path, size):
            if not path or not os.path.exists(path): return None
            img = load_image(path, size)
            return img.unsqueeze(0).cpu() if img is not None else None # BHWC float

        depth_map_bhwc = _load_aux(depth_path, target_size)
        canny_map_bhwc = _load_aux(canny_path, target_size)
        sam_map_bhwc = _load_aux(sam_path, target_size) # This will be thresholded inside execute

        if sam_map_bhwc is not None: print(f"Loaded SAM map: {sam_map_bhwc.shape}")
        if depth_map_bhwc is not None: print(f"Loaded Depth map: {depth_map_bhwc.shape}")
        if canny_map_bhwc is not None: print(f"Loaded Canny map: {canny_map_bhwc.shape}")

        # --- 3. Visualize Inputs ---
        visualize_inputs(
            output_dir=output_dir,
            test_name=test_name,
            input_img=(input_img_bhwc[0] * 255).byte(), # Pass HWC uint8
            depth_map=(depth_map_bhwc[0] * 255).byte() if depth_map_bhwc is not None else None,
            canny_map=(canny_map_bhwc[0] * 255).byte() if canny_map_bhwc is not None else None
        )

        # --- 4. Prepare Node Parameters ---
        node_params = {
            "input_image": input_img_bhwc, # BHWC float
            "SAM_map": sam_map_bhwc,       # BHWC float or None
            "depth_map": depth_map_bhwc,   # BHWC float or None
            "canny_map": canny_map_bhwc,   # BHWC float or None
            "max_steps": test_config.get("max_steps", 500),
            "start_field": test_config.get("start_field", StartField.CENTER.value),
            "pulsate_strength": test_config.get("pulsate_strength", 0.0),
            "directional_flow": test_config.get("directional_flow", False),
            "branch_awareness": test_config.get("branch_awareness", 0.0),
            "growth_threshold": test_config.get("growth_threshold", 0.6),
            "seed_radius": test_config.get("seed_radius", 5),
            "stability_threshold": test_config.get("stability_threshold", 20),
            "noise_low": test_config.get("noise_low", 0.0),
            "noise_high": test_config.get("noise_high", 1.2),
            "lab_gradient_scale": test_config.get("lab_gradient_scale", 5.0),
        }

        # --- 5. Run Organic Fill Node ---
        fill_node = OrganicFillNode()
        # Returns final_mask (BHW float), frames_preview (NHWC float for batch 0)
        final_mask_bhw, frames_preview_nhwc = fill_node.execute(**node_params)

        # --- 6. Visualize & Save Outputs ---
        # Utilities expect HW uint8 mask, NHW uint8 frames, HWC uint8 input

        # Get first batch item for mask, convert HW float -> HW byte
        final_mask_hw_byte = (final_mask_bhw[0] * 255).byte().cpu()
        save_final_mask(output_dir, test_name, final_mask_hw_byte)

        # Convert frames preview NHWC float -> NHW byte
        frames_nhw_byte = (frames_preview_nhwc[..., 0] * 255).byte().cpu() # Take first channel
        save_frames_as_gif(frames_nhw_byte, os.path.join(output_dir, f"{test_name}_fill_process.gif"))
        visualize_frames_grid(output_dir, test_name, frames_nhw_byte)

        # Input image HWC uint8, final mask HW byte
        input_img_hwc_byte = (input_img_bhwc[0] * 255).byte().cpu()
        visualize_final_result(
            output_dir=output_dir,
            test_name=test_name,
            input_img=input_img_hwc_byte,
            final_mask=final_mask_hw_byte
        )

        print(f"--- Test {test_name} complete. Results saved to {output_dir}/ ---")


    # --- Define Test Setup ---
    script_dir = os.path.dirname(__file__)
    test_dir = os.path.join(script_dir, "test_assets")
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Define Test Cases ---
    test_cases = [
        {   # Test 1: Only Input Image (should use LAB gradient)
            "name": "img2_depth_only",
            "input": os.path.join(test_dir, "img2.jpg"),
            "depth": os.path.join(test_dir, "img2_depth.jpg"),
            "max_steps": 300,
            "start_field": StartField.CENTER.value,
            "lab_gradient_scale": 10.0,
        },
        {   # Test 2: Input + Depth + Canny
            "name": "img2_depth_canny",
            "input": os.path.join(test_dir, "img2.jpg"),
            "depth": os.path.join(test_dir, "img2_depth.jpg"),
            "canny": os.path.join(test_dir, "img2_canny.jpg"),
            "start_field": StartField.CLOSEST_POINT.value, # Use depth minimum
            "directional_flow": True,
            "branch_awareness": 0.3, # Should be ignored as no SAM mask -> full base mask
            "max_steps": 500
        }
    ]

    # --- Run Tests ---
    for config in test_cases:
        run_test_case(config, output_dir)

    print("\nAll specified tests completed!")
