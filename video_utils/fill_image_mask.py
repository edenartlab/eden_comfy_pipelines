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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ────────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────

_SOBEL_X = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]).view(1,1,3,3)
_SOBEL_Y = torch.tensor([[1., 2.,  1.], [0., 0.,  0.], [-1., -2., -1.]]).view(1,1,3,3)

def _sobel_grad(img: torch.Tensor) -> torch.Tensor:
    """Return gradient magnitude of a single‑channel image."""
    if img.dim() == 4:
        img = img.permute(0,3,1,2)  # [B,1,H,W]
    grad_x = F.conv2d(img, _SOBEL_X.to(img.device), padding=1)
    grad_y = F.conv2d(img, _SOBEL_Y.to(img.device), padding=1)
    mag = torch.sqrt(grad_x**2 + grad_y**2)
    return mag.squeeze(1)  # [B,H,W]

def _normalise(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mn, mx = t.min(), t.max()
    # Avoid division by zero if min and max are the same
    if mx - mn < eps:
        return torch.zeros_like(t)
    return (t - mn) / (mx - mn + eps)


# ────────────────────────────────────────────────────────────────────────────────
#  CORE ORGANIC‑FILL IMPLEMENTATION (BATCH‑AWARE)
# ────────────────────────────────────────────────────────────────────────────────

class OrganicFillBatch:
    """Performs confidence‑aware organic fill on a batch of images."""
    def __init__(self,
                 base_mask: torch.Tensor,  # bool, [B,H,W]
                 depth: Optional[torch.Tensor] = None,  # [B,H,W] float 0‑1
                 canny: Optional[torch.Tensor] = None,  # [B,H,W] float/bool
                 sem: Optional[torch.Tensor] = None,    # [B,H,W] float 0‑1
                 start_field: StartField = StartField.RANDOM,
                 config: FillNodeConfig = None):

        self.cfg = config or FillNodeConfig()
        self.device = torch.device(self.cfg.device)

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
        """Compute confidence contributions as described in the spec."""
        B, H, W = self.B, self.H, self.W
        device = self.device

        P_sem = self.sem if self.sem is not None else torch.ones((B,H,W), device=device)
        P_sem = _normalise(P_sem)

        # Signed distance field (inside +, outside ‑)
        # We need distance OUTSIDE mask to penalise leaks.
        print("Computing distance transform...")
        sdf_out = []
        for b in range(B):
            m_sk = tensor_to_np_mask(self.mask[b]>0)
            dist_out = cv2.distanceTransform(255-m_sk, cv2.DIST_L2, 5)
            sdf_out.append(torch.from_numpy(dist_out))
        SDF_OUT = torch.stack(sdf_out).to(device)  # [B,H,W]
        SDF_OUT = _normalise(SDF_OUT)

        # Depth gradient penalty
        if self.depth is not None:
            print("Computing depth gradients...")
            d_norm = _normalise(self.depth)
            grad_mag = _sobel_grad(d_norm.unsqueeze(-1))  # [B,H,W]
            depth_term = torch.exp(-grad_mag*10)
        else:
            depth_term = torch.ones((B,H,W), device=device)

        # Edge penalty (prefer pixels away from strong canny edges)
        if self.canny is not None:
            E = 1.0 - _normalise(self.canny.float())
        else:
            E = torch.ones((B,H,W), device=device)

        # Combine – linear fusion then clamp to [0,1]
        self.confidence = torch.clamp(
            0.55*P_sem  - 0.25*torch.tanh(SDF_OUT*3) + 0.25*depth_term, 0.0, 1.0
        ) * E

        # Branch awareness – compute skeleton distance once
        if self.cfg.branch_awareness > 0.0:
            print("Computing branch awareness maps...")
            try:
                from skimage.morphology import skeletonize
            except ImportError:
                print("Warning: skimage not found, branch awareness disabled")
                self.branch_weight = torch.ones((B,H,W), device=device)
                return
            bw = []
            for b in range(B):
                # Use the imported utility function
                sk = skeletonize((self.mask[b]>0).cpu().numpy()).astype(np.uint8)
                dist_to_skel = cv2.distanceTransform(255-sk*255, cv2.DIST_L2, 3)
                # Normalize distance robustly
                max_dist = dist_to_skel.max()
                dist_norm = dist_to_skel / (max_dist + 1e-8) if max_dist > 0 else dist_to_skel
                bw.append(1.0 - torch.from_numpy(dist_norm))  # 1 near skeleton, 0 far
            self.branch_weight = torch.stack(bw).to(device)
        else:
            self.branch_weight = torch.ones((B,H,W), device=device)

    # ────────────────────────────────────────────────────────────────────────
    #  SEED PLACEMENT
    # ────────────────────────────────────────────────────────────────────────
    def _place_seeds(self, start_field: StartField):
        B,H,W = self.B,self.H,self.W
        y_coord = torch.arange(H, device=self.device).view(1,H,1).expand(B,H,W)
        x_coord = torch.arange(W, device=self.device).view(1,1,W).expand(B,H,W)
        valid = self.mask>0

        def pick_seed(mask_b, selector):
            idx = torch.nonzero(mask_b & selector)
            if idx.numel() == 0:  # fallback – random inside mask
                idx = torch.nonzero(mask_b)
            sel = idx[torch.randint(len(idx),(1,))]
            return sel

        print(f"Placing seeds using {start_field.value} strategy...")
        for b in range(B):
            m = valid[b]
            # Simplify seed point selection logic
            if start_field == StartField.CLOSEST_POINT and self.depth is not None:
                depth_b = self.depth[b]
                # Use torch.where for cleaner masking
                depth_inside = torch.where(m, depth_b, torch.full_like(depth_b, float('inf')))
                min_depth = depth_inside.min()
                # Handle cases where no valid depth minimum exists within the mask
                yx_candidates = torch.nonzero(depth_inside == min_depth, as_tuple=False)
                if yx_candidates.numel() > 0:
                    yx = yx_candidates[0] # Take the first one if multiple minima
                else: # Fallback if no min depth found or depth map is weird
                     yx_valid = torch.nonzero(m, as_tuple=False)
                     if yx_valid.numel() > 0:
                         yx = yx_valid[torch.randint(len(yx_valid),(1,))]
                     else: # Should not happen if mask is valid, but handle edge case
                         print(f"Warning: No valid pixels found in mask for batch {b}. Placing seed at center.")
                         yx = torch.tensor([[H//2, W//2]], device=self.device)
                y, x = yx[0], yx[1]
            else:
                # Simplified region selection
                h2, w2 = H // 2, W // 2
                thirds_h, thirds_w = H // 3, W // 3
                sel = torch.zeros_like(m)
                if start_field == StartField.RANDOM:
                    sel = m # Select from anywhere inside mask
                elif start_field == StartField.TOP_LEFT:      sel[:h2, :w2] = 1
                elif start_field == StartField.TOP_RIGHT:     sel[:h2, w2:] = 1
                elif start_field == StartField.BOTTOM_LEFT:   sel[h2:, :w2] = 1
                elif start_field == StartField.BOTTOM_RIGHT:  sel[h2:, w2:] = 1
                elif start_field == StartField.CENTER:        sel[thirds_h:2*thirds_h, thirds_w:2*thirds_w] = 1
                elif start_field == StartField.TOP_CENTER:    sel[:thirds_h, thirds_w:2*thirds_w] = 1
                elif start_field == StartField.BOTTOM_CENTER: sel[2*thirds_h:, thirds_w:2*thirds_w] = 1
                elif start_field == StartField.LEFT_CENTER:   sel[thirds_h:2*thirds_h, :thirds_w] = 1
                elif start_field == StartField.RIGHT_CENTER:  sel[thirds_h:2*thirds_h, 2*thirds_w:] = 1

                # Find valid points within the selected region and the mask
                yx_candidates = torch.nonzero(m & sel.bool(), as_tuple=False)
                if yx_candidates.numel() == 0: # Fallback to any valid point in mask
                    yx_candidates = torch.nonzero(m, as_tuple=False)

                if yx_candidates.numel() > 0:
                    yx = yx_candidates[torch.randint(len(yx_candidates), (1,))]
                    y, x = yx[0,0], yx[0,1]
                else: # Should not happen if mask is valid
                    print(f"Warning: No valid pixels found for seeding in batch {b}. Placing seed at center.")
                    y, x = H//2, W//2

            # Apply circular seed
            yy, xx = torch.meshgrid(torch.arange(H,device=self.device), torch.arange(W,device=self.device), indexing='ij')
            circle = ((yy-y)**2 + (xx-x)**2 <= self.cfg.seed_radius**2)
            self.grid[b][circle & (self.mask[b]>0)] = 1.0

    # ────────────────────────────────────────────────────────────────────────
    #  CORE STEP
    # ────────────────────────────────────────────────────────────────────────
    def _update_activity(self, new_growth: torch.Tensor):
        self.activity_counter[new_growth] = 0
        self.activity_counter[~new_growth] += 1
        recent = self.activity_counter < self.cfg.stability_threshold
        
        # Use GPU for dilate operation if available
        if hasattr(torch, 'nn') and hasattr(torch.nn, 'functional') and hasattr(torch.nn.functional, 'max_pool2d'):
            # Use max_pool2d for dilation on GPU
            padding = self.cfg.active_region_padding
            padded = F.max_pool2d(recent.float().unsqueeze(1), 
                                 kernel_size=2*padding+1, 
                                 stride=1, 
                                 padding=padding).squeeze(1) > 0
        else:
            # Fallback to OpenCV (ensure numpy array is contiguous)
            recent_np = np.ascontiguousarray(recent.cpu().numpy().astype(np.uint8))
            kernel = np.ones((2*self.cfg.active_region_padding+1,)*2, np.uint8)
            dilated_np = cv2.dilate(recent_np, kernel)
            padded = torch.from_numpy(dilated_np).bool().to(self.device)
            
        self.active_mask = padded & (self.grid<1) & (self.mask>0)

    def step(self):
        self.frame_count += 1
        B,H,W = self.B,self.H,self.W
        padded = F.pad(self.grid, (1,1,1,1))
        neighbours = F.unfold(padded.unsqueeze(1), kernel_size=3).view(B,9,H,W)
        nbr_hit = (neighbours.sum(1) - neighbours[:,4]) > 0

        # Growth probability = confidence * branch_weight
        P = self.confidence * ((1-self.cfg.branch_awareness) + self.cfg.branch_awareness * self.branch_weight)

        # Directional flow via depth gradient
        if self.cfg.directional_flow and self.depth is not None:
            grad_mag = _sobel_grad(_normalise(self.depth).unsqueeze(-1))
            P = P * (1.0 + grad_mag)
            P = torch.clamp(P,0,1)

        # Pulsating threshold
        thr_mod = self.cfg.growth_threshold + self.cfg.pulsate_strength * math.sin(2*math.pi*self.frame_count/60)

        boundary = (self.grid==0) & (self.mask>0) & nbr_hit & self.active_mask & (P>thr_mod)
        noise = torch.rand_like(self.grid, device=self.device)*(self.cfg.noise_high-self.cfg.noise_low)+self.cfg.noise_low
        new_growth = boundary & (noise < P)
        self.grid[new_growth]=1.0
        self._update_activity(new_growth)
        
        # Calculate fill ratio and track history
        fill_ratio = self.fill_ratio()
        self.fill_history.append(fill_ratio)
        
        # Return active pixel count for progress reporting
        active_count = torch.sum(self.active_mask).item()
        return active_count, fill_ratio.mean().item()

    # ────────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────────────────────────
    def fill_ratio(self):
        return (self.grid*self.mask).sum(dim=(1,2)) / (self.mask.sum(dim=(1,2)) + 1e-8)

    def is_complete(self):
        if len(self.fill_history) < self.cfg.saturation_window:
            return False
        recent_change = torch.max(torch.stack([
            torch.abs(self.fill_history[i] - self.fill_history[i-1])
            for i in range(-self.cfg.saturation_window+1,0)
        ]))
        return (recent_change < self.cfg.saturation_threshold) and not torch.any((self.grid==0)&self.active_mask&(self.mask>0))

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
                "input_image": ("IMAGE", {}),  # RGB or Gray
            },
            "optional": {
                "depth_map": ("IMAGE", {}),   # assumed single channel
                "canny_map": ("IMAGE", {}),   # Changed from MASK to IMAGE
                "SAM_map": ("IMAGE", {}),
                "start_field": ( [sf.value for sf in StartField], {"default": StartField.CENTER.value} ),
                "pulsate_strength": ("FLOAT", {"default": 0.0, "min":0.0, "max":1.0, "step":0.01}),
                "directional_flow": ("BOOLEAN", {"default": False}),
                "branch_awareness": ("FLOAT", {"default": 0.0, "min":0.0, "max":1.0, "step":0.01}),
                "max_steps": ("INT", {"default":2000, "min":1, "max":10000}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK")  # final mask, list of frames
    RETURN_NAMES = ("final_mask", "frames")

    FUNCTION = "execute"
    CATEGORY = "Eden 🌱"

    # ────────────────────────────────────────────────────────────────────────
    def execute(self,
                input_image: torch.Tensor,
                depth_map: Optional[torch.Tensor] = None,
                canny_map: Optional[torch.Tensor] = None,
                SAM_map: Optional[torch.Tensor] = None,
                start_field: str = StartField.CENTER.value,
                pulsate_strength: float = 0.0,
                directional_flow: bool = False,
                branch_awareness: float = 0.0,
                max_steps: int = 2000
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"OrganicFill starting on {device} device with max {max_steps} steps")
        print(f"Input image dimensions: {input_image.shape}")

        # Convert image to mask base via threshold if SAM not provided
        if SAM_map is not None:
            base_mask = (SAM_map[...,0] > 0.5).float()  # assume first channel prob
        else:
            # simple luminance threshold on input_image
            if input_image.shape[-1] == 3:
                gray = (0.2126*input_image[...,0] + 0.7152*input_image[...,1] + 0.0722*input_image[...,2])
            else:
                gray = input_image[...,0]
            base_mask = (gray < gray.mean()).float()  # heuristic

        if base_mask.dim()==2:
            base_mask = base_mask.unsqueeze(0)  # [1,H,W]
        elif base_mask.dim()==3:
            base_mask = base_mask  # [B,H,W]
        else:  # [B,H,W,1]
            base_mask = base_mask.squeeze(-1)

        print(f"Base mask dimensions: {base_mask.shape}")

        # Ensure auxiliary maps shaped [B,H,W]
        def _prep_aux(aux):
            if aux is None: return None
            # Fix dimension handling
            if aux.dim() == 4 and aux.shape[-1] == 3:  # [B,H,W,3]
                # RGB? Convert to grayscale
                aux = 0.299*aux[...,0] + 0.587*aux[...,1] + 0.114*aux[...,2]
            elif aux.dim() == 4 and aux.shape[-1] == 1:  # [B,H,W,1]
                aux = aux.squeeze(-1)
            elif aux.dim() == 3 and aux.shape[-1] == 3:  # [H,W,3]
                # Single RGB image - convert to grayscale
                aux = 0.299*aux[...,0] + 0.587*aux[...,1] + 0.114*aux[...,2]
                aux = aux.unsqueeze(0)  # Add batch dimension
            elif aux.dim() == 3 and aux.shape[-1] == 1:  # [H,W,1]
                # Single grayscale image
                aux = aux.squeeze(-1).unsqueeze(0)  # Remove channel, add batch
            
            # Ensure we always return a 3D tensor [B,H,W]
            if aux.dim() == 2:  # [H,W]
                aux = aux.unsqueeze(0)
            
            # Make sure it's on the correct device
            return aux.to(device)
        
        print("Preparing auxiliary maps...")
        if depth_map is not None:
            depth_map = _prep_aux(depth_map)
            print(f"Processed depth map dimensions: {depth_map.shape}")
        
        if canny_map is not None:
            canny_map = _prep_aux(canny_map)
            print(f"Processed canny map dimensions: {canny_map.shape}")
        
        if SAM_map is not None:
            SAM_prob = _prep_aux(SAM_map)
            print(f"Processed SAM map dimensions: {SAM_prob.shape}")
        else:
            SAM_prob = None

        # Print image dimensions
        print(f"Image dimensions: {base_mask.shape}")
        
        cfg = FillNodeConfig(pulsate_strength=pulsate_strength,
                             directional_flow=directional_flow,
                             branch_awareness=branch_awareness,
                             device=device)

        fill = OrganicFillBatch(base_mask=base_mask,
                                depth=depth_map,
                                canny=canny_map,
                                sem=SAM_prob,
                                start_field=StartField(start_field),
                                config=cfg)

        frames: List[torch.Tensor] = []
        step = 0
        
        print("Starting organic fill process...")
        progress_interval = max(1, min(100, max_steps // 20))  # Report progress at most 20 times
        
        while not fill.is_complete() and step < max_steps:
            active_pixels, fill_ratio = fill.step()
            
            # Store frames at appropriate intervals or important moments
            if step < 20 or step % 20 == 0 or step >= max_steps-5 or fill.is_complete():
                frames.append(fill.get_frame().clone())
            
            # Print progress updates
            if step % progress_interval == 0 or step == max_steps-1 or fill.is_complete():
                elapsed = time.time() - start_time
                print(f"Step {step}/{max_steps} | Fill ratio: {fill_ratio:.3f} | Active pixels: {active_pixels} | Elapsed: {elapsed:.2f}s")
            
            step += 1
            
        # Add final frame if not already added
        if len(frames) == 0 or not torch.all(frames[-1] == fill.get_frame()):
            frames.append(fill.get_frame().clone())
            
        final_mask = fill.get_frame()
        total_time = time.time() - start_time
        print(f"Organic fill completed in {total_time:.2f}s after {step} steps")

        # stack the frames into a single tensor
        frames = torch.stack(frames)
        
        return final_mask, frames


# ────────────────────────────────────────────────────────────────────────────────
#  TEST FUNCTIONALITY
# ────────────────────────────────────────────────────────────────────────────────

def run_test_case(test_config: Dict[str, Any], output_dir: str):
    """Loads data, runs the fill node, and saves visualizations for a single test case."""
    test_name = test_config.get("name", "unnamed_test")
    print(f"\n--- Running test case: {test_name} ---")

    # --- 1. Load Input Image ---
    input_path = test_config.get("input")
    if not input_path:
        print(f"Skipping {test_name}: 'input' path missing in config.")
        return
    input_img_hwc = load_image(input_path)
    if input_img_hwc is None:
        print(f"Skipping {test_name}: Failed to load input image at '{input_path}'.")
        return
    input_img_bhwc = input_img_hwc.unsqueeze(0).cpu()
    target_size = input_img_hwc.shape[:2]  # (H, W)
    H, W = target_size
    print(f"Input image loaded ({test_name}): {input_img_bhwc.shape}, Target size: {target_size}")

    # --- 2. Load Auxiliary Maps ---
    depth_path = test_config.get("depth")
    canny_path = test_config.get("canny")
    sam_path = test_config.get("SAM") # Use "SAM" key for consistency

    depth_map_hwc = load_image(depth_path, target_size) if depth_path else None
    canny_map_hwc = load_image(canny_path, target_size) if canny_path else None
    sam_map_hwc = load_image(sam_path, target_size) if sam_path else None

    depth_map_bhwc = depth_map_hwc.unsqueeze(0).cpu() if depth_map_hwc is not None else None
    canny_map_bhwc = canny_map_hwc.unsqueeze(0).cpu() if canny_map_hwc is not None else None

    # --- 3. Prepare Mask (Load SAM or Create Synthetically) ---
    if sam_map_hwc is not None:
        # Assume loaded SAM map is already a mask or probability map [0,1]
        # Convert to BHWC, ensure single channel
        if sam_map_hwc.shape[-1] > 1: # Take first channel if multi-channel
            sam_map_hwc = sam_map_hwc[..., 0:1]
        sam_map_bhwc = sam_map_hwc.unsqueeze(0).cpu()
        print(f"Using SAM map for {test_name}: {sam_map_bhwc.shape}")
    else:
        sam_map_bhwc = None

    # --- 4. Visualize Inputs ---
    visualize_inputs(
        output_dir=output_dir,
        test_name=test_name,
        input_img=input_img_bhwc[0], # Pass HWC
        depth_map=depth_map_bhwc[0] if depth_map_bhwc is not None else None,
        canny_map=canny_map_bhwc[0] if canny_map_bhwc is not None else None
    )

    # --- 5. Prepare Node Parameters ---
    node_params = {
        "input_image": input_img_bhwc,
        "depth_map": depth_map_bhwc,
        "canny_map": canny_map_bhwc,
        "SAM_map": sam_map_bhwc, # Provide the prepared mask/sam map here
        "max_steps": test_config.get("max_steps", 500),
        "start_field": test_config.get("start_field", StartField.CENTER.value),
        "pulsate_strength": test_config.get("pulsate_strength", 0.1),
        "directional_flow": test_config.get("directional_flow", True),
        "branch_awareness": test_config.get("branch_awareness", 0.3),
    }

    # --- 6. Run Organic Fill Node ---
    fill_node = OrganicFillNode()
    final_mask, frames = fill_node.execute(**node_params) # Returns BHW, N H W

    # Ensure shapes are correct for saving utilities (expect HW for mask, NHW for frames)
    if final_mask.dim() == 3: # BHW -> HW
        final_mask_hw = final_mask[0]
    else: # Assume already HW
        final_mask_hw = final_mask

    if frames.dim() == 4: # NBH -> NHW
        frames_nhw = frames[:, 0, :, :]
    else: # Assume already NHW
        frames_nhw = frames

    # Save final mask (pass HW)
    save_final_mask(output_dir, test_name, final_mask_hw)

    # Save animated fill process (pass NHW)
    save_frames_as_gif(frames_nhw, os.path.join(output_dir, f"{test_name}_fill_process.gif"))

    # Visualize frame grid (pass NHW)
    visualize_frames_grid(output_dir, test_name, frames_nhw)

    # Visualize final result blended (pass HWC input, HW mask)
    visualize_final_result(
        output_dir=output_dir,
        test_name=test_name,
        input_img=input_img_bhwc[0], # Pass HWC
        final_mask=final_mask_hw    # Pass HW
    )

    print(f"--- Test {test_name} complete. Results saved to {output_dir}/ ---")


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend for saving files

    # Define test assets paths (relative to script location)
    script_dir = os.path.dirname(__file__)
    test_dir = os.path.join(script_dir, "test_assets")
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # Define test cases as dictionaries
    test_cases = [
        {
            "name": "img1_depth",
            "input": os.path.join(test_dir, "img2.jpg"),
            "depth": os.path.join(test_dir, "img2_depth.jpg"),
            "start_field": StartField.CENTER.value,
            "pulsate_strength": 0.2,
            "directional_flow": False,
            "branch_awareness": 0.1,
            "max_steps": 600
        },
        {
            "name": "img1_depth_canny",
            "input": os.path.join(test_dir, "img2.jpg"),
            "depth": os.path.join(test_dir, "img2_depth.jpg"),
            "canny": os.path.join(test_dir, "img2_canny.jpg"),
            "start_field": StartField.CLOSEST_POINT.value, # Use depth minimum
            "pulsate_strength": 0.0,
            "directional_flow": True,
            "branch_awareness": 0.5,
            "max_steps": 700
        }
    ]

    for config in test_cases:
        run_test_case(config, output_dir)

    print("\nAll specified tests completed!")
