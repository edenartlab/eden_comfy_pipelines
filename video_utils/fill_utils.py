import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from typing import List, Tuple, Optional

# ────────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ────────────────────────────────────────────────────────────────────────────────

def pil_to_comfy_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor format [B,H,W,C]."""
    # Convert PIL to tensor [C,H,W] with values in [0,1]
    tensor = ToTensor()(pil_image)
    # Permute to [H,W,C]
    tensor = tensor.permute(1, 2, 0)
    # Add batch dimension [B,H,W,C]
    tensor = tensor.unsqueeze(0)
    return tensor

def comfy_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor format [B,H,W,C] to PIL Image."""
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # [H,W,C]
    # Ensure tensor is on CPU and in float format
    tensor = tensor.cpu().float()
    # Clamp to [0,1] range
    tensor = torch.clamp(tensor, 0, 1)
    # Permute to [C,H,W] for ToPILImage
    tensor = tensor.permute(2, 0, 1)
    # Convert to PIL
    return ToPILImage()(tensor)

def tensor_to_np_mask(t: torch.Tensor) -> np.ndarray:
    """Convert boolean Torch mask [H,W] to uint8 0/255 numpy."""
    # Ensure input is on CPU before converting to numpy
    if t.device != torch.device('cpu'):
        t = t.cpu()
    return (t.numpy().astype(np.uint8) * 255)

# Helper function for normalization
def normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor to [0, 1] range.
    
    For tensors with dim < 3: Normalizes the entire tensor.
    For tensors with dim >= 3: Normalizes each batch element independently.
    
    Args:
        tensor: Input tensor to normalize
        eps: Small epsilon value to avoid division by zero
        
    Returns:
        Normalized tensor in [0, 1] range
    """
    if tensor is None:
        return None
        
    if tensor.numel() == 0:
        return tensor  # Handle empty tensor
    
    if tensor.dim() < 3:  # Handle cases like [H,W] or single values
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        if max_val - min_val < eps:
            return torch.zeros_like(tensor)
        return (tensor - min_val) / (max_val - min_val + eps)
    else:  # Handle batched tensors [B, H, W] or [B, C, H, W], etc.
        # Normalize each element in the batch independently
        tensor_norm = torch.zeros_like(tensor, dtype=torch.float32)
        for i in range(tensor.shape[0]):
            min_val = tensor[i].min()
            max_val = tensor[i].max()
            if max_val - min_val < eps:
                tensor_norm[i] = torch.zeros_like(tensor[i])
            else:
                tensor_norm[i] = (tensor[i] - min_val) / (max_val - min_val + eps)
        return tensor_norm

def load_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> Optional[torch.Tensor]:
    """Load image from path and convert to tensor [H,W,C] with optional resizing."""
    img = Image.open(path)

    # Ensure image is in RGB or L mode before processing
    if img.mode not in ['RGB', 'L', 'RGBA']:
        img = img.convert('RGB')
    elif img.mode == 'RGBA':
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        img = background

    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BICUBIC)

    tensor = ToTensor()(img) # [C,H,W]

    # Convert grayscale (L mode) or single-channel RGB to [1, H, W]
    if tensor.shape[0] == 1:
        pass # Already [1, H, W]
    elif tensor.shape[0] == 3 and torch.allclose(tensor[0], tensor[1], atol=1e-3) and torch.allclose(tensor[1], tensor[2], atol=1e-3):
        tensor = tensor[0:1] # Take one channel -> [1, H, W]
    elif tensor.shape[0] == 4: # RGBA converted by ToTensor()
         tensor = tensor[:3] # Drop alpha -> [3, H, W]
         
    return tensor.float().permute(1, 2, 0) # [H,W,C]


def save_frames_as_gif(frames: torch.Tensor, output_path: str, duration: int = 100):
    """Save frames tensor [N,H,W], [N,1,H,W] or [N,H,W,1] as animated gif."""
    if frames.dim() == 4 and frames.shape[1] == 1:
        frames = frames.squeeze(1) # [N, H, W]
    elif frames.dim() == 4 and frames.shape[-1] == 1:
         frames = frames.squeeze(-1) # [N, H, W]
    elif frames.dim() != 3:
         raise ValueError(f"Expected frames tensor of shape [N,H,W], [N,1,H,W] or [N,H,W,1], got {frames.shape}")

    # Ensure tensor is on CPU and float type [0, 1] range for ToPILImage
    frames = frames.cpu().float()

    # Normalize each frame individually to [0, 1]
    normalized_frames = [normalize_tensor(f) for f in frames]

    frames_pil = [ToPILImage()(f) for f in normalized_frames]
    if frames_pil:
        try:
            frames_pil[0].save(
                output_path,
                save_all=True,
                append_images=frames_pil[1:],
                optimize=True,
                duration=duration,
                loop=0
            )
            print(f"- Fill process animation: {os.path.basename(output_path)}")
        except Exception as e:
            print(f"Error saving GIF {output_path}: {e}")
    else:
        print(f"Warning: No frames provided to save_frames_as_gif for {output_path}")


# ────────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION (for testing/debugging)
# ────────────────────────────────────────────────────────────────────────────────

def visualize_inputs(output_dir: str, test_name: str, input_img: Optional[torch.Tensor], depth_map: Optional[torch.Tensor], canny_map: Optional[torch.Tensor]):
    """Save a visualization of the input images and masks."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{test_name}_inputs.png")

    num_plots = sum(x is not None for x in [input_img, depth_map, canny_map])
    if num_plots == 0:
        print("No inputs to visualize.")
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    axes = [axes] if num_plots == 1 else axes.flatten() # Handle single plot case
    plot_idx = 0

    def plot_img(ax, img_tensor, title, cmap=None):
        if img_tensor is None: return False
        # Convert to numpy first, as imshow handles uint8 [0, 255] correctly
        img_np = img_tensor.cpu().numpy()

        # Handle different channel configurations [H,W,C], [H,W], [C,H,W]
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3]: # [C,H,W] -> [H,W,C]
            img_np = np.transpose(img_np, (1, 2, 0))
        if img_np.ndim == 3 and img_np.shape[-1] == 1: # [H,W,1] -> [H,W]
            img_np = img_np.squeeze(-1)

        ax.imshow(img_np, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        return True

    if plot_idx < num_plots and plot_img(axes[plot_idx], input_img, "Input Image"): plot_idx += 1
    if plot_idx < num_plots and plot_img(axes[plot_idx], depth_map, "Depth Map", cmap='viridis'): plot_idx += 1
    if plot_idx < num_plots and plot_img(axes[plot_idx], canny_map, "Canny Edges", cmap='gray'): plot_idx += 1

    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print(f"- Input visualization: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"Error saving input visualization {output_path}: {e}")
    plt.close()


def visualize_frames_grid(output_dir: str, test_name: str, frames: torch.Tensor, num_frames_to_show: int = 9):
    """Save a grid visualization of selected frames from the fill process."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{test_name}_frames_grid.png")

    if frames.dim() == 4 and frames.shape[1] == 1:
        frames = frames.squeeze(1) # [N, H, W]
    elif frames.dim() == 4 and frames.shape[-1] == 1:
         frames = frames.squeeze(-1) # [N, H, W]

    if frames.dim() != 3:
        print(f"Warning: Invalid frames shape {frames.shape} for visualize_frames_grid.")
        return

    frames = frames.cpu().float() # Ensure CPU and float for plotting
    # No normalization needed here if imshow handles ranges okay, or if frames are known to be [0,1]
    # Let's assume frames are okay based on user feedback for now.

    if len(frames) == 0:
        print(f"Warning: No frames provided to visualize_frames_grid for {test_name}")
        return

    num_frames_actual = min(num_frames_to_show, len(frames))
    if num_frames_actual == 0: return # Avoid division by zero if len(frames) is 0
    frame_indices = torch.linspace(0, len(frames)-1, num_frames_actual).long()

    ncols = 3
    nrows = (num_frames_actual + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False) # squeeze=False ensures 2D array
    axes = axes.flatten()

    for i, idx in enumerate(frame_indices):
        ax = axes[i]
        # Normalize frame before displaying
        frame_to_plot = normalize_tensor(frames[idx])
        ax.imshow(frame_to_plot.numpy(), cmap='gray')
        ax.set_title(f"Frame {idx}")
        ax.axis('off')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print(f"- Frames grid: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"Error saving frames grid {output_path}: {e}")
    plt.close()


def visualize_final_result(output_dir: str, test_name: str, input_img: Optional[torch.Tensor], final_mask: torch.Tensor):
    """Save a visualization comparing the original image, final mask, and blended result."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{test_name}_result_visualization.png")

    # --- Prepare Tensors ---
    # Normalize first, then ensure CPU float
    final_mask_norm = normalize_tensor(final_mask).cpu().float()
    input_img_norm = None
    if input_img is not None:
        input_img_norm = normalize_tensor(input_img).cpu().float()


    # Standardize mask: [H, W]
    if final_mask_norm.dim() == 4 and final_mask_norm.shape[1] == 1: final_mask_norm = final_mask_norm.squeeze(1) #[B, H, W]
    if final_mask_norm.dim() == 4 and final_mask_norm.shape[-1] == 1: final_mask_norm = final_mask_norm.squeeze(-1) #[B, H, W]
    if final_mask_norm.dim() == 3 and final_mask_norm.shape[0] == 1: final_mask_norm = final_mask_norm.squeeze(0) # [H, W]
    if final_mask_norm.dim() != 2:
        print(f"Warning: Unexpected final_mask shape {final_mask_norm.shape}, skipping visualization.")
        return

    # Standardize input image: [H, W, C] or None
    if input_img_norm is not None:
        if input_img_norm.dim() == 4 and input_img_norm.shape[0] == 1: input_img_norm = input_img_norm.squeeze(0) # [C, H, W] or [H, W, C]
        if input_img_norm.dim() == 3 and input_img_norm.shape[0] in [1, 3]: # [C, H, W] -> [H, W, C]
             input_img_norm = input_img_norm.permute(1, 2, 0)
        if input_img_norm.dim() != 3 or input_img_norm.shape[-1] not in [1, 3]:
            print(f"Warning: Unexpected input_img shape {input_img_norm.shape}, cannot blend.")
            input_img_norm = None # Treat as unavailable for blending


    # --- Plotting ---
    num_plots = 2 + (input_img_norm is not None) # Original, Mask, Blend (if possible)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    axes = [axes] if num_plots == 1 else axes.flatten()
    plot_idx = 0

    # Plot Original Image (if available)
    if input_img_norm is not None:
        input_np = input_img_norm.numpy()
        axes[plot_idx].imshow(input_np.squeeze()) # Squeeze for grayscale [H,W,1]
        axes[plot_idx].set_title("Original Image")
        axes[plot_idx].axis('off')
        plot_idx += 1
    else:
        # If input image is missing, occupy the first slot with a message
        axes[plot_idx].text(0.5, 0.5, 'Original Image Not Available', ha='center', va='center')
        axes[plot_idx].axis('off')
        plot_idx += 1


    # Plot Final Mask
    mask_np = final_mask_norm.numpy()
    axes[plot_idx].imshow(mask_np, cmap='gray')
    axes[plot_idx].set_title("Final Mask")
    axes[plot_idx].axis('off')
    plot_idx += 1

    # Plot Blended Result (if original image available)
    if input_img_norm is not None:
        input_np = input_img_norm.numpy()
        # Create RGB mask broadcastable to image shape
        mask_rgb = np.stack([mask_np]*input_np.shape[-1], axis=-1) if input_np.ndim == 3 else mask_np # Handle grayscale input

        # Ensure input_np is compatible (e.g., grayscale input with RGB mask?)
        if input_np.shape[-1] == 1 and mask_rgb.shape[-1] == 3:
             input_np_rgb = np.repeat(input_np, 3, axis=-1)
        else:
             input_np_rgb = input_np

        # Blend if shapes match
        if input_np_rgb.shape == mask_rgb.shape:
            blended = 0.7 * input_np_rgb + 0.3 * mask_rgb
            axes[plot_idx].imshow(np.clip(blended, 0, 1))
            axes[plot_idx].set_title("Blended Result")
        else:
            axes[plot_idx].set_title("Blend Error (Shape Mismatch)")
        axes[plot_idx].axis('off')
        plot_idx += 1


    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print(f"- Result visualization: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"Error saving result visualization {output_path}: {e}")
    plt.close()


def save_final_mask(output_dir: str, test_name: str, final_mask: torch.Tensor):
    """Save the final generated mask as a PNG image."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{test_name}_final_mask.png")

    # Standardize mask: [H, W] before saving
    if final_mask.dim() == 4 and final_mask.shape[1] == 1: final_mask = final_mask.squeeze(1) #[B, H, W]
    if final_mask.dim() == 4 and final_mask.shape[-1] == 1: final_mask = final_mask.squeeze(-1) #[B, H, W]
    if final_mask.dim() == 3 and final_mask.shape[0] == 1: final_mask = final_mask.squeeze(0) # [H, W]

    if final_mask.dim() != 2:
        print(f"Warning: Cannot save final mask with shape {final_mask.shape}.")
        return

    try:
        # Ensure tensor is on CPU and in a suitable format for ToPILImage (e.g., float [0,1] or uint8)
        # Normalize before converting
        final_mask_norm = normalize_tensor(final_mask).cpu().float()
        final_pil = ToPILImage()(final_mask_norm)
        final_pil.save(output_path)
        print(f"- Final mask: {os.path.basename(output_path)}")
    except Exception as e:
         print(f"Error saving final mask {output_path}: {e}") 