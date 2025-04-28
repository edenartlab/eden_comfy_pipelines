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
    return (t - mn) / (mx - mn + eps)


def _tensor_to_np_mask(t: torch.Tensor) -> np.ndarray:
    """Convert boolean Torch mask [H,W] to uint8 0/255 numpy."""
    return (t.cpu().numpy().astype(np.uint8) * 255)


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
            m_sk = _tensor_to_np_mask(self.mask[b]>0)
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
                sk = skeletonize((self.mask[b]>0).cpu().numpy()).astype(np.uint8)
                dist_to_skel = cv2.distanceTransform(255-sk*255, cv2.DIST_L2, 3)
                dist_norm = dist_to_skel / (dist_to_skel.max()+1e-8)
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
            if start_field == StartField.CLOSEST_POINT and self.depth is not None:
                depth_b = self.depth[b]
                depth_inside = torch.where(m, depth_b, torch.full_like(depth_b, 1e9))
                yx = torch.nonzero(depth_inside == depth_inside.min())
                if yx.numel()==0:
                    yx = torch.nonzero(m)
                y, x = yx[0]
            else:
                if start_field == StartField.RANDOM:
                    y, x = pick_seed(m, m)[0]
                else:
                    h2, w2 = H//2, W//2
                    thirds_h, thirds_w = H//3, W//3
                    sel = torch.zeros_like(m)
                    # Build selector logically
                    if start_field == StartField.TOP_LEFT:
                        sel[:h2,:w2]=1
                    elif start_field == StartField.TOP_RIGHT:
                        sel[:h2,w2:]=1
                    elif start_field == StartField.BOTTOM_LEFT:
                        sel[h2:,:w2]=1
                    elif start_field == StartField.BOTTOM_RIGHT:
                        sel[h2:,w2:]=1
                    elif start_field == StartField.CENTER:
                        sel[h2//2:h2+h2//2, w2//2:w2+w2//2]=1
                    elif start_field == StartField.TOP_CENTER:
                        sel[:thirds_h, thirds_w:2*thirds_w]=1
                    elif start_field == StartField.BOTTOM_CENTER:
                        sel[2*thirds_h:, thirds_w:2*thirds_w]=1
                    elif start_field == StartField.LEFT_CENTER:
                        sel[thirds_h:2*thirds_h, :thirds_w]=1
                    elif start_field == StartField.RIGHT_CENTER:
                        sel[thirds_h:2*thirds_h, 2*thirds_w:]=1
                    else:  # default random
                        sel = m
                    y, x = pick_seed(m, sel.bool())[0]
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
            # Fallback to OpenCV
            recent_np = recent.cpu().numpy().astype(np.uint8)
            kernel = np.ones((2*self.cfg.active_region_padding+1,)*2, np.uint8)
            padded = torch.from_numpy(cv2.dilate(recent_np, kernel)).bool().to(self.device)
            
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
            print(f"Input depth map dimensions: {depth_map.shape}")
            depth_map = _prep_aux(depth_map)
            print(f"Processed depth map dimensions: {depth_map.shape}")
        
        if canny_map is not None:
            print(f"Input canny map dimensions: {canny_map.shape}")
            canny_map = _prep_aux(canny_map)
            print(f"Processed canny map dimensions: {canny_map.shape}")
        
        if SAM_map is not None:
            print(f"Input SAM map dimensions: {SAM_map.shape}")
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

if __name__ == "__main__":
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    from torchvision.transforms import ToTensor, ToPILImage, Resize

    def load_image(path, target_size=None):
        """Load image from path and convert to tensor with optional resizing."""
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return None
        img = Image.open(path)
        
        # Resize if target_size is provided
        if target_size is not None:
            # Use BICUBIC instead of deprecated LANCZOS
            img = img.resize((target_size[1], target_size[0]), Image.BICUBIC)
        
        # Convert to tensor and handle channels
        tensor = ToTensor()(img).permute(1, 2, 0)  # [H,W,C]
        
        # If grayscale image saved as RGB (3 identical channels), convert to single channel
        if tensor.shape[-1] == 3 and torch.allclose(tensor[...,0], tensor[...,1], atol=1e-3) and torch.allclose(tensor[...,1], tensor[...,2], atol=1e-3):
            tensor = tensor[...,0:1]
            
        return tensor

    def save_frames_as_gif(frames, output_path):
        """Save frames as animated gif."""
        frames_pil = [ToPILImage()(f.squeeze(0)) for f in frames]
        frames_pil[0].save(
            output_path, 
            save_all=True,
            append_images=frames_pil[1:],
            optimize=True,
            duration=100,
            loop=0
        )

    # Test assets paths
    test_dir = "test_assets"
    test_cases = [
        {
            "name": "img1",
            "input": os.path.join(test_dir, "img1.jpg"),
            "depth": os.path.join(test_dir, "img1_depth.png"),
            "canny": os.path.join(test_dir, "img1_canny.png"),
        },
        {
            "name": "img2",
            "input": os.path.join(test_dir, "img2.jpg"),
            "depth": os.path.join(test_dir, "img2_depth.png"),
            "canny": os.path.join(test_dir, "img2_canny.png"),
        }
    ]

    # Create output directory if it doesn't exist
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)

    # Run tests
    for test in test_cases:
        print(f"\n\nTesting on {test['name']}...")
        
        # Load input image first to get dimensions
        input_img = load_image(test["input"])
        
        if input_img is None:
            print(f"Skipping {test['name']} due to missing input image")
            continue
            
        # Get target size from input image
        target_size = input_img.shape[:2]  # (H, W)
        print(f"Using target size: {target_size}")
        
        # Load auxiliary maps with resizing to match input dimensions
        depth_map = load_image(test["depth"], target_size)
        canny_map = load_image(test["canny"], target_size)
        
        # Ensure depth is single-channel
        if depth_map is not None and depth_map.shape[-1] == 3:
            # Convert to grayscale if it's RGB
            depth_map = 0.299 * depth_map[...,0] + 0.587 * depth_map[...,1] + 0.114 * depth_map[...,2]
            depth_map = depth_map.unsqueeze(-1)  # Add channel dimension back
            
        # Same for canny
        if canny_map is not None and canny_map.shape[-1] == 3:
            # Convert to grayscale if it's RGB
            canny_map = 0.299 * canny_map[...,0] + 0.587 * canny_map[...,1] + 0.114 * canny_map[...,2]
            canny_map = canny_map.unsqueeze(-1)  # Add channel dimension back
        
        # Create directory for outputs
        os.makedirs("test_output", exist_ok=True)
        
        # Create a synthetic mask for testing (circle in center)
        H, W = target_size
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        
        # Use different shapes for the two test cases
        if test["name"] == "img1":
            # Create an ellipse for img1
            circle_mask = (((y - H//2)**2)/(H//6)**2 + ((x - W//2)**2)/(W//6)**2 < 1).float()
        else:
            # Create two circles for img2
            circle1 = ((y - H//3)**2 + (x - W//3)**2 < min(H,W)**2//25).float()
            circle2 = ((y - 2*H//3)**2 + (x - 2*W//3)**2 < min(H,W)**2//20).float()
            circle_mask = torch.clamp(circle1 + circle2, 0, 1)
        
        circle_mask = circle_mask.unsqueeze(-1)
        
        # Save the synthetic mask
        debug_mask = ToPILImage()(circle_mask.squeeze(-1))
        debug_mask.save(f"test_output/{test['name']}_seed_mask.png")
        
        # Create a composite visualization of the input
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(input_img.cpu().numpy())
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        axes[1].imshow(depth_map.squeeze(-1).cpu().numpy(), cmap='viridis')
        axes[1].set_title("Depth Map")
        axes[1].axis('off')
        
        axes[2].imshow(canny_map.squeeze(-1).cpu().numpy(), cmap='gray')
        axes[2].set_title("Canny Edges")
        axes[2].axis('off')
        
        axes[3].imshow(circle_mask.squeeze(-1).cpu().numpy(), cmap='gray')
        axes[3].set_title("Seed Mask")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"test_output/{test['name']}_inputs.png")
        plt.close()
        
        # Run organic fill with the synthetic mask and more steps
        fill_node = OrganicFillNode()
        final_mask, frames = fill_node.execute(
            input_image=input_img,
            depth_map=depth_map,
            canny_map=canny_map,
            SAM_map=circle_mask,  # Use our synthetic mask
            max_steps=500,  # More steps to allow better filling
            start_field=StartField.CENTER.value,
            pulsate_strength=0.1,
            directional_flow=True,
            branch_awareness=0.3
        )
        
        # Save results
        output_base = os.path.join(output_dir, test["name"])
        
        # Save final mask
        final_pil = ToPILImage()(final_mask.squeeze(0))
        final_pil.save(f"{output_base}_final_mask.png")
        
        # Save animated fill process
        save_frames_as_gif(frames, f"{output_base}_fill_process.gif")
        
        # Display intermediate frames as grid
        num_frames = min(9, len(frames))
        frame_indices = torch.linspace(0, len(frames)-1, num_frames).long()
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, idx in enumerate(frame_indices):
            ax = axes[i//3, i%3]
            ax.imshow(frames[idx].squeeze(0).cpu().numpy(), cmap='gray')
            ax.set_title(f"Frame {idx}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_base}_frames_grid.png")
        plt.close()
        
        # Create a visualization of the final result blended with the original image
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(input_img.cpu().numpy())
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Final mask
        axes[1].imshow(final_mask.squeeze(0).cpu().numpy(), cmap='gray')
        axes[1].set_title("Final Mask")
        axes[1].axis('off')
        
        # Blended result - properly broadcast mask to match image dimensions
        mask_np = final_mask.squeeze(0).cpu().numpy()
        input_np = input_img.cpu().numpy()
        
        # Create RGB mask with same dimensions as input
        mask_rgb = np.zeros_like(input_np)
        for i in range(3):  # Copy mask to all 3 channels
            mask_rgb[..., i] = mask_np
            
        # Blend images
        blended = 0.7 * input_np + 0.3 * mask_rgb
        
        axes[2].imshow(blended)
        axes[2].set_title("Blended Result")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_base}_result_visualization.png")
        plt.close()
        
        print(f"Saved results for {test['name']} to {output_dir}/")
        print(f"- Final mask: {test['name']}_final_mask.png")
        print(f"- Fill process animation: {test['name']}_fill_process.gif")
        print(f"- Frames grid: {test['name']}_frames_grid.png")
        print(f"- Result visualization: {test['name']}_result_visualization.png")
    
    print("\nAll tests completed!")
