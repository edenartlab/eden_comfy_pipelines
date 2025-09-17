import os, time, sys
import cv2
import numpy as np
import random
import gc
import torch
import imghdr

from PIL import Image, ImageOps, ImageSequence
import torch.nn.functional as F

from scipy import ndimage

import torchvision.transforms.functional as T
import comfy.utils
from torch.cuda.amp import autocast
import psutil

###########################################################################

# Import comfyUI modules:
from cli_args import args
import folder_paths

###########################################################################

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

def print_available_memory():
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / 1024 / 1024 / 1024:.2f} GB")

class Eden_RGBA_to_RGB:
    """Node that converts RGBA images to RGB by alpha blending with a background color"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "background_color": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    FUNCTION = "convert_rgba_to_rgb"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    CATEGORY = "Eden 🌱/transform"

    def convert_rgba_to_rgb(self, images, background_color=0.0):
        """
        Convert RGBA images to RGB by alpha blending with background color
        
        Args:
            images: Input tensor [B,H,W,C] where C can be 3 or 4, values in range 0-1
            background_color: Background color value (0-1)
            
        Returns:
            RGB images tensor [B,H,W,3] with values in range 0-1
        """
        # Return early if tensor is empty
        if images is None or images.numel() == 0:
            return (images,)
            
        batch_size, height, width, channels = images.shape
        
        # Create output tensor for RGB images
        if channels == 4:
            result = torch.zeros(batch_size, height, width, 3, dtype=images.dtype, device=images.device)
        else:
            result = images.clone()
            
        # Process each image in the batch
        for b in range(batch_size):
            img = images[b]
            
            if channels == 4:  # Has alpha channel
                # Extract RGB and alpha channels
                rgb = img[:, :, :3].float()
                alpha = img[:, :, 3].float()  # Alpha already in 0-1 range
                
                # Alpha blend with background color
                # Formula: result = alpha * foreground + (1 - alpha) * background
                background = torch.full_like(rgb, background_color)
                blended = alpha.unsqueeze(2) * rgb + (1 - alpha.unsqueeze(2)) * background
                
                # Keep in 0-1 range
                result[b] = torch.clamp(blended, 0, 1)
            else:
                # Already RGB or other format, keep as is
                result[b] = img[:, :, :3] if img.shape[2] >= 3 else img
                
        return (result,)

class Eden_Random_Flip:
    """Node that randomly flips each image or mask in a batch horizontally with a given probability"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Accept any tensor type (IMAGE, MASK, etc.)
                "flip_probability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    FUNCTION = "random_flip"
    RETURN_TYPES = ("IMAGE",)  # Return the same type as input
    RETURN_NAMES = ("images",)
    CATEGORY = "Eden 🌱/transform"

    def random_flip(self, tensor, flip_probability=0.5):
        """
        Randomly flip each image/mask in a batch horizontally with the given probability
        
        Args:
            tensor: Input tensor, can be IMAGE [B,H,W,C] or MASK [B,H,W]
            flip_probability: Probability of applying the horizontal flip for each individual image
            
        Returns:
            Tensor of the same type, with individual images possibly flipped horizontally
        """
        # Return early if tensor is empty
        if tensor is None or tensor.numel() == 0:
            return tensor
            
        # Create a copy of the tensor to avoid modifying the original
        result = tensor.clone()
        
        # Determine tensor format
        if len(tensor.shape) == 4 and tensor.shape[3] in [1, 3, 4]:
            # IMAGE format [B,H,W,C]
            batch_size = tensor.shape[0]
            for b in range(batch_size):
                if random.random() < flip_probability:
                    # Flip this individual image along width dimension (dim=1)
                    result[b] = torch.flip(tensor[b], dims=[1])
                    
        elif len(tensor.shape) == 3:
            # MASK format [B,H,W]
            batch_size = tensor.shape[0]
            for b in range(batch_size):
                if random.random() < flip_probability:
                    # Flip this individual mask along width dimension (dim=1)
                    result[b] = torch.flip(tensor[b], dims=[1])
                    
        elif len(tensor.shape) == 4 and tensor.shape[1] in [1, 3, 4]:
            # [B,C,H,W] format
            batch_size = tensor.shape[0]
            for b in range(batch_size):
                if random.random() < flip_probability:
                    # Flip this individual image along width dimension (dim=2)
                    result[b] = torch.flip(tensor[b], dims=[2])
                    
        else:
            # If format is unknown, treat first dimension as batch
            # and assume second-to-last dimension is width
            batch_size = tensor.shape[0]
            for b in range(batch_size):
                if random.random() < flip_probability:
                    # Flip along the second-to-last dimension
                    flip_dims = [len(tensor.shape) - 2]
                    result[b] = torch.flip(tensor[b], dims=flip_dims)
                    
        return result

class Eden_MaskBoundingBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
                "noise_threshold": ("INT", { "default": 1, "min": 0, "max": 1000, "step": 1, }),
            },
            "optional": {
                "image_optional": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("MASK", "IMAGE", "x", "y", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "Eden/mask"

    def execute(self, mask, padding, blur, noise_threshold, image_optional=None):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if image_optional is None:
            image_optional = mask.unsqueeze(3).repeat(1, 1, 1, 3)
        # resize the image if it's not the same size as the mask
        if image_optional.shape[1:] != mask.shape[1:]:
            image_optional = comfy.utils.common_upscale(image_optional.permute([0,3,1,2]), mask.shape[2], mask.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])
        # match batch size
        if image_optional.shape[0] < mask.shape[0]:
            image_optional = torch.cat((image_optional, image_optional[-1].unsqueeze(0).repeat(mask.shape[0]-image_optional.shape[0], 1, 1, 1)), dim=0)
        elif image_optional.shape[0] > mask.shape[0]:
            image_optional = image_optional[:mask.shape[0]]
        
        # Apply noise reduction
        kernel_size = 3
        mask = self.reduce_noise(mask, kernel_size, noise_threshold)

        # blur the mask
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            mask = T.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1)
        
        # Find bounding box
        y_indices, x_indices = torch.where(mask[0] > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x1 = max(0, x_indices.min().item() - padding)
            x2 = min(mask.shape[2], x_indices.max().item() + 1 + padding)
            y1 = max(0, y_indices.min().item() - padding)
            y2 = min(mask.shape[1], y_indices.max().item() + 1 + padding)
        else:
            # If no non-zero pixels found, return the entire mask
            x1, y1, x2, y2 = 0, 0, mask.shape[2], mask.shape[1]

        # crop the mask and debug_mask
        mask = mask[:, y1:y2, x1:x2]
        image_optional = image_optional[:, y1:y2, x1:x2, :]

        return (mask, image_optional, x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def reduce_noise(mask, kernel_size, threshold):
        # Create a max pooling layer
        max_pool = torch.nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
        
        # Apply max pooling
        pooled = max_pool(mask)
        
        # Count non-zero neighbors
        neighbor_count = torch.nn.functional.conv2d(
            mask.float(), 
            torch.ones(1, 1, kernel_size, kernel_size).to(mask.device),
            padding=kernel_size//2
        )
        
        # Keep only pixels with enough non-zero neighbors
        mask = torch.where(neighbor_count >= threshold, pooled, torch.zeros_like(pooled))
        
        return mask


class WidthHeightPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"width":  ("INT", {"default": 512, "min": 0, "max": sys.maxsize}),
                     "height":  ("INT", {"default": 512, "min": 0, "max": sys.maxsize}),
                     "output_multiplier":  ("FLOAT", {"default": 0.5}),
                     "multiple_off":  ("INT", {"default": 64, "min": 1, "max": 264}),
                     }
                }

    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width","height",)
    FUNCTION = "compute_resolution"
    OUTPUT_NODE = False
    CATEGORY = "Eden 🌱"

    def compute_resolution(self, width, height, output_multiplier, multiple_off):
        width = int(width * output_multiplier)
        height = int(height * output_multiplier)

        # round to closest multiple of multiple_off:
        width = int(round(width / multiple_off) * multiple_off)
        height = int(round(height / multiple_off) * multiple_off)

        print(f"Using final resolution: width x height = {width} x {height}")

        return width, height





class SaveImageAdvanced:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "add_timestamp": ("BOOLEAN", {"default": True}),
                     "save_metadata_json": ("BOOLEAN", {"default": True}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Eden 🌱"

    def save_images(self, images, add_timestamp, save_metadata_json, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()

        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(full_output_folder, exist_ok = True)
        
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                metadata_dict = {}
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                    metadata_dict["prompt"] = prompt
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                        metadata_dict[x] = extra_pnginfo[x]

            if add_timestamp:
                file = f"{filename_prefix}_{timestamp_str}_{counter:05}.png"
            else:
                file = f"{filename_prefix}_{counter:05}_.png"
            
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })

            if save_metadata_json and not args.disable_metadata:
                json_path = os.path.join(full_output_folder, file.replace(".png", ".json"))
                with open(json_path, "w") as f:
                    json.dump(metadata_dict, f, indent=4)                

            counter += 1

        return { "ui": { "images": results } }




class LatentTypeConversion:
    """
    Allows storing latents in float16 format to save memory, and converting to float32 when needed.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", ),
                "output_type": (["float16", "float32"], ),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "convert"
    CATEGORY = "Eden 🌱"

    def convert(self, latent, output_type="float16", verbose=True):
        if verbose:
            print_available_memory()
            print(f"Input latent type: {latent['samples'].dtype}")
            print(f"Input latent shape: {latent['samples'].shape}")
            print(f"Input device: {latent['samples'].device}")

        # Ensure the tensor is on the correct device (GPU if available)
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #latent["samples"] = latent["samples"].to(device)

        # Use autocast for automatic type conversion in mixed precision settings
        with autocast():
            if output_type == "float32" and latent["samples"].dtype == torch.float16:
                latent["samples"] = latent["samples"].float()
            elif output_type == "float16" and latent["samples"].dtype == torch.float32:
                latent["samples"] = latent["samples"].half()

        if verbose:
            print(f"After conversion, latent type: {latent['samples'].dtype}")
            print_available_memory()

        return (latent,)

class VAEDecode_to_folder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                 "samples": ("LATENT", ), 
                 "vae": ("VAE", ),
                 "prefix": ("STRING", {"default": "test"}),
                 "output_folder": ("STRING", {"default": "output/frames"}),
                }
            }
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "decode"
    CATEGORY = "Eden 🌱"

    def decode(self, vae, samples, prefix, output_folder):
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        output_folder = os.path.join(output_folder, f"{prefix}_{timestamp_str}")
        os.makedirs(output_folder, exist_ok=True)

        for i, sample in enumerate(samples["samples"]):
            img = vae.decode(sample.unsqueeze(0))
            img = img.cpu().numpy() * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img.squeeze())
            img.save(os.path.join(output_folder, f"{i:06d}.jpg"), quality=95)

        return (output_folder, )



from sklearn.cluster import KMeans
from .img_utils import lab_to_rgb, rgb_to_lab
import numpy as np
from PIL.PngImagePlugin import PngInfo
import json
import torch
import torch.nn.functional as F

def gaussian_kernel_2d(kernel_size, sigma=None):
    """Create a 2D Gaussian kernel with proper size handling."""
    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    # Create 1D kernels
    kernel_range = torch.arange(-(kernel_size//2), kernel_size//2+1)
    kernel_1d = torch.exp(-(kernel_range**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()  # Ensure normalization
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    return kernel_2d

class MaskFromRGB_KMeans:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "n_color_clusters": ("INT", {"default": 6, "min": 2, "max": 10}),
                "clustering_resolution": ("INT", {"default": 256, "min": 32, "max": 1024}),
                "feathering_fraction": ("FLOAT", { "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01 }),
                "equalize_areas": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("1","2","3","4","5","6","7","8","combined",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱"

    @torch.no_grad()
    def execute(self, image, n_color_clusters, clustering_resolution, feathering_fraction, equalize_areas):
        # Get appropriate device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("--- Warning: Using MPS backend, ensure your PyTorch version supports it.")
            device = torch.device('mps')
        else:
            print("--- Warning: No GPU available for running MaskfromRGB_Kmeans, using CPU...")
            device = torch.device('cpu')
            
        # Store original device
        original_device = image.device
        
        # Move image to the appropriate device
        image = image.to(device)
        
        # Convert to LAB color space
        lab_images = torch.stack([rgb_to_lab(img) for img in image])
        n, h, w, _ = lab_images.shape

        # Bring channel dim to second position
        lab_images = lab_images.permute(0, 3, 1, 2)

        # Maintain aspect ratio
        h_target, w_target = clustering_resolution, int(clustering_resolution * w / h)
        lab_images = F.interpolate(lab_images, size=[h_target, w_target], mode='bicubic', align_corners=False)
        
        # Bring channel dim back to last position
        lab_images = lab_images.permute(0, 2, 3, 1)

        # Reshape images for k-means clustering
        n, h, w, _ = lab_images.shape
        lab_images_reshaped = lab_images.view(n*w*h, 3)

        # Move to CPU for KMeans (sklearn doesn't support GPU)
        lab_images_cpu = lab_images_reshaped.cpu().numpy()
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_color_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(lab_images_cpu)
        
        # Calculate average luminance for each cluster
        cluster_centers = kmeans.cluster_centers_
        cluster_luminance = cluster_centers[:, 0]  # L channel in LAB color space
        
        # Sort cluster indices based on luminance
        sorted_indices = np.argsort(cluster_luminance)
        index_map = {old: new for new, old in enumerate(sorted_indices)}
        
        # Map the cluster labels to new sorted indices
        sorted_cluster_labels = np.vectorize(index_map.get)(cluster_labels)
        cluster_labels = torch.from_numpy(sorted_cluster_labels).to(device).view(n, h, w)
        
        # Apply area equalization if requested
        if equalize_areas > 0:
            # Count the frequency of each cluster (across all frames)
            cluster_counts = torch.zeros(n_color_clusters, device=device)
            for i in range(n_color_clusters):
                cluster_counts[i] = (cluster_labels == i).sum().float()
            
            # Calculate the average target count if clusters were equal
            avg_count = cluster_counts.sum() / n_color_clusters
            
            # Calculate the percentile threshold for each cluster based on counts
            thresholds = []
            
            # Convert to numpy for easier processing
            cluster_labels_cpu = cluster_labels.cpu().numpy()
            
            # Create a flattened view for convenient processing
            flat_labels = cluster_labels_cpu.reshape(-1)
            
            # Process each cluster
            for i in range(n_color_clusters):
                # Get the flattened LAB values for this cluster
                mask = flat_labels == i
                cluster_size = mask.sum()
                
                if cluster_size == 0:
                    # Skip empty clusters
                    thresholds.append(None)
                    continue
                
                cluster_lab_values = lab_images_cpu[mask]
                
                # Calculate distances to the cluster center
                distances = np.linalg.norm(cluster_lab_values - cluster_centers[sorted_indices[i]], axis=1)
                
                # Sort distances
                sorted_distances = np.sort(distances)
                
                # Calculate the current to target ratio
                current_size = cluster_counts[i].item()
                ratio = current_size / avg_count.item()
                
                # Apply the equalize_areas weight
                weighted_ratio = 1.0 + (ratio - 1.0) * equalize_areas
                
                # Calculate the target size after weighting
                target_size = current_size / weighted_ratio
                
                # If target size is smaller, we need a threshold to cut off the furthest points
                if target_size < current_size:
                    percentile = (target_size / current_size) * 100
                    threshold_idx = min(len(sorted_distances) - 1, int(percentile * len(sorted_distances) / 100))
                    thresholds.append(sorted_distances[threshold_idx])
                else:
                    # No threshold needed if we're expanding
                    thresholds.append(None)
            
            # Create a new tensor for the adjusted labels
            adjusted_labels = cluster_labels.clone()
            
            # Process each cluster
            for i in range(n_color_clusters):
                if thresholds[i] is not None:
                    # Get the LAB values for this cluster
                    cluster_mask = (cluster_labels == i)
                    
                    if not cluster_mask.any():
                        continue
                    
                    # Calculate distances for all pixels in this cluster
                    cluster_data = lab_images.view(-1, 3)[cluster_mask.view(-1)]
                    cluster_center = torch.tensor(cluster_centers[sorted_indices[i]], device=device)
                    
                    # Calculate squared distances to the cluster center
                    distances = torch.sum((cluster_data - cluster_center)**2, dim=1)
                    
                    # Create a mask for points to reassign
                    reassign_mask = distances > thresholds[i]**2
                    
                    if reassign_mask.any():
                        # Find the indices in the original tensor
                        indices = torch.nonzero(cluster_mask.view(-1), as_tuple=True)[0][reassign_mask]
                        
                        # For these points, find the next closest cluster
                        points_to_reassign = lab_images.view(-1, 3)[indices]
                        
                        # Calculate distances to all cluster centers
                        all_distances = torch.zeros((len(points_to_reassign), n_color_clusters), device=device)
                        
                        for j in range(n_color_clusters):
                            if j == i:
                                # Set distance to current cluster as infinity to force reassignment
                                all_distances[:, j] = float('inf')
                            else:
                                center = torch.tensor(cluster_centers[sorted_indices[j]], device=device)
                                all_distances[:, j] = torch.sum((points_to_reassign - center)**2, dim=1)
                        
                        # Find the closest cluster for each point
                        new_clusters = torch.argmin(all_distances, dim=1)
                        
                        # Update the adjusted labels
                        flat_adjusted = adjusted_labels.view(-1)
                        flat_adjusted[indices] = new_clusters
            
            # Replace the original labels with the adjusted ones
            cluster_labels = adjusted_labels
        
        # Transform the cluster_labels into masks
        masks = torch.zeros(n, 8, h, w, device=device)

        for i in range(n):
            for j in range(min(n_color_clusters, 8)):
                masks[i, j] = (cluster_labels[i] == j).float()

        # Create the combined mask
        combined_mask = torch.zeros(n, h, w, device=device)
        for i in range(n):
            for j in range(n_color_clusters):
                # Map each cluster to a shade of gray (0 to 1)
                if j < n_color_clusters:
                    gray_value = j / (n_color_clusters - 1) if n_color_clusters > 1 else 0
                    combined_mask[i] += (cluster_labels[i] == j).float() * gray_value

        if feathering_fraction > 0:
            n_imgs, n_colors, h, w = masks.shape
            batch_size = n_imgs * n_colors
            masks = masks.view(batch_size, h, w)

            # Calculate feathering size
            feathering = int(feathering_fraction * (w+h)/2)
            
            # Ensure kernel size is appropriate (odd and not too large)
            feathering = max(3, feathering)
            if feathering % 2 == 0:
                feathering += 1
                
            print(f"Using feathering kernel size: {feathering}")
            
            # Create the kernel on the appropriate device
            kernel = gaussian_kernel_2d(feathering).to(device)
            
            # Calculate padding size
            pad_size = feathering // 2
            
            print("Feathering masks...")
            # Apply convolution for feathering
            masks_feathered = torch.zeros_like(masks)
            
            for i in range(masks.shape[0]):
                mask_padded = masks[i].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                
                # Apply reflection padding
                mask_padded = F.pad(mask_padded, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
                
                # Apply convolution
                mask_feathered = F.conv2d(mask_padded, kernel, padding=0)
                
                # Store the result
                masks_feathered[i] = mask_feathered.squeeze()
            
            masks = masks_feathered.view(n_imgs, n_colors, h, w)
            
            # Also feather the combined mask
            combined_mask_padded = combined_mask.unsqueeze(1)  # Add channel dimension
            combined_mask_batch = combined_mask_padded.view(n, 1, h, w)
            
            combined_mask_feathered = torch.zeros_like(combined_mask_batch)
            
            for i in range(combined_mask_batch.shape[0]):
                mask_padded = combined_mask_batch[i].unsqueeze(0)  # Add batch dimension
                
                # Apply reflection padding
                mask_padded = F.pad(mask_padded, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
                
                # Apply convolution
                mask_feathered = F.conv2d(mask_padded, kernel, padding=0)
                
                # Store the result
                combined_mask_feathered[i] = mask_feathered.squeeze()
            
            combined_mask = combined_mask_feathered.squeeze(1)

        # Upscale masks to original resolution
        masks = F.interpolate(masks, size=(image.shape[1], image.shape[2]), mode='bicubic', align_corners=False)
        combined_mask = F.interpolate(combined_mask.unsqueeze(1), size=(image.shape[1], image.shape[2]), mode='bicubic', align_corners=False).squeeze(1)
        
        # Move masks back to the original device
        masks = masks.to(original_device)
        combined_mask = combined_mask.to(original_device)

        return masks[:, 0], masks[:, 1], masks[:, 2], masks[:, 3], masks[:, 4], masks[:, 5], masks[:, 6], masks[:, 7], combined_mask


class Eden_MaskCombiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_a": ("MASK",),
                "rel_strength_a": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01}),
                "rel_strength_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "rel_strength_c": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "lower_clamp": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "upper_clamp": ("FLOAT", {"default": 98.0, "min": 50.0, "max": 100.0, "step": 0.5}),
                "gamma": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.05})
            },
            "optional": {
                "mask_b": ("MASK", {"default": None}),
                "mask_c": ("MASK", {"default": None})
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine_masks"
    CATEGORY = "Eden 🌱"

    def soft_clamp(self, x, min_val, max_val, smoothness=0.1):
        """Apply soft clamping using sigmoid function in PyTorch"""
        range_val = max_val - min_val
        normalized = (x - min_val) / range_val
        return torch.sigmoid((normalized - 0.5) / smoothness)

    def adaptive_histogram_eq(self, img, num_bins=256):
        """Apply adaptive histogram equalization using PyTorch"""
        # Ensure input is in correct range
        img = torch.clamp(img, 0, 1)
        
        # Calculate histogram
        hist = torch.histc(img, bins=num_bins, min=0, max=1)
        
        # Calculate CDF
        cdf = torch.cumsum(hist, dim=0)
        cdf_normalized = cdf * hist.min() / cdf.max()
        
        # Create bin centers
        bin_centers = torch.linspace(0, 1, num_bins, device=img.device)
        
        # Interpolate values
        img_flat = img.reshape(-1)
        indices = torch.bucketize(img_flat, bin_centers)
        indices = torch.clamp(indices, 0, num_bins - 1)
        
        return cdf_normalized[indices].reshape(img.shape)

    def compute_quantile(self, tensor, q, max_elements = 10000):
        """Compute quantile for large tensors by sampling"""
        # If tensor is too large, sample it
        if tensor.numel() > max_elements:
            indices = torch.randperm(tensor.numel(), device=tensor.device)[:max_elements]
            tensor_sample = tensor.view(-1)[indices]
            return torch.quantile(tensor_sample, q)
        else:
            return torch.quantile(tensor, q)

    @torch.no_grad()
    def combine_masks(self, mask_a, rel_strength_a, lower_clamp, upper_clamp, gamma,
                     mask_b=None, mask_c=None, rel_strength_b=None, rel_strength_c=None):
        """
        ComfyUI node function to combine multiple masks with improved signal preservation.
        All operations are performed in PyTorch.
        """
        device = mask_a.device

        # Apply gamma correction
        mask_a = torch.pow(mask_a, gamma)
        masks = [mask_a]
        weights = [rel_strength_a]
        
        # Process optional masks
        if mask_b is not None and rel_strength_b != 0:
            mask_b = torch.pow(mask_b, gamma)
            masks.append(mask_b)
            weights.append(rel_strength_b)
            
        if mask_c is not None and rel_strength_c != 0:
            mask_c = torch.pow(mask_c, gamma)
            masks.append(mask_c)
            weights.append(rel_strength_c)
        
        # Process masks with strength values
        processed_masks = []
        processed_weights = []
        
        for mask, strength in zip(masks, weights):
            if strength != 0:
                if strength < 0:
                    processed_masks.append(1 - mask)
                    processed_weights.append(abs(strength))
                else:
                    processed_masks.append(mask)
                    processed_weights.append(strength)
        
        # If no valid masks, return mid-gray
        if not processed_masks:
            return (torch.full_like(mask_a, 0.5),)
        
        # Normalize weights using softmax
        weights_tensor = torch.tensor(processed_weights, device=device)
        weights_tensor = F.softmax(weights_tensor, dim=0)
        
        # Combine masks
        combined = torch.zeros_like(mask_a)
        for mask, weight in zip(processed_masks, weights_tensor):
            combined += mask * weight

        # Apply adaptive histogram equalization
        #combined = self.adaptive_histogram_eq(combined)
        
        # Calculate and apply adaptive clamp thresholds
        lower_threshold = self.compute_quantile(combined, lower_clamp/100)
        upper_threshold = self.compute_quantile(combined, upper_clamp/100)
        combined = self.soft_clamp(combined, lower_threshold, upper_threshold)
        
        # Apply inverse gamma correction
        combined = torch.pow(combined, 1/gamma)
        
        return (combined,)

def convert_pnginfo_to_dict(pnginfo: PngInfo) -> dict:
    """
    Convert a PngInfo object to a Python dictionary.

    :param pnginfo: PngInfo object to be converted.
    :return: A dictionary representation of the PngInfo object.
    """
    if not isinstance(pnginfo, PngInfo):
        raise TypeError("Expected a PngInfo object")

    metadata_dict = {}
    # Accessing the internal structure of PngInfo object
    for key in pnginfo.__dict__.get("text", {}):
        metadata_dict[key] = pnginfo.get(key)

    return metadata_dict

def detect_faces(image):
    import mediapipe as mp
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Convert PIL image to numpy array if it's not already
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Create a copy of the image for drawing
    image_copy = img_array.copy()
    
    # Create a black and white mask image
    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    
    # Using MediaPipe face detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # Convert the image to RGB format as MediaPipe requires RGB input
        rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
        
        # Process the image with MediaPipe
        results = face_detection.process(rgb_image)
        
        face_count = 0
        
        # Check if any faces were detected
        if results.detections:
            face_count = len(results.detections)
            for detection in results.detections:
                # Get the bounding box
                bboxC = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to pixel coordinates
                h, w = img_array.shape[:2]
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Ensure coordinates are within image boundaries
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Draw rectangle on the image copy
                cv2.rectangle(image_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Set the corresponding region in the mask image to white
                mask[y:y+height, x:x+width] = 255
        
        print('Faces Detected:', face_count)
        
        # Convert the mask to a PIL Image
        mask_pil = Image.fromarray(mask)
        
        return mask_pil

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Eden_FaceToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",)},
                }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "Eden 🌱/face"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)
    
    def run(self, image):
        im = tensor2pil(image)
        mask = detect_faces(im)
        mask = pil2tensor(mask)
        return (mask,)

class Eden_Face_Crop:
    """Takes a hard or soft mask image, crops out the main face and returns reconstruction mask and face images."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI format: [B, H, W, C]
                "face_mask": ("MASK",),  # ComfyUI format: [B, H, W]
                "padding_factor": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_face_ratio": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 1.0, "step": 0.001}),
            }
        }

    FUNCTION = "crop_face"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "MASK", "MASK")
    RETURN_NAMES = ("cropped_face", "crop_x", "crop_y", "crop_width", "crop_height", "crop_mask", "custom_mask")
    CATEGORY = "Eden 🌱/face"
    
    def find_main_face_bbox(self, mask_tensor, threshold=0.5, min_face_ratio=0.01):
        """Find the bounding box of the main face region"""
        # Validate inputs
        if mask_tensor is None or mask_tensor.numel() == 0:
            return None, None
            
        # Convert mask to numpy - keep the 2D structure
        try:
            # Handle ComfyUI mask format [B, H, W]
            if len(mask_tensor.shape) == 3:  # [B, H, W]
                mask_np = mask_tensor[0].cpu().numpy()
            else:
                mask_np = mask_tensor.cpu().numpy()
                
            # Validate mask after conversion
            if mask_np.size == 0:
                return None, None
                
        except Exception as e:
            return None, None
        
        # Normalize mask to 0-1 if needed
        if mask_np.max() > 1.0 + 1e-6:  # Add small epsilon for floating point comparison
            mask_np = mask_np / 255.0
        
        # Binary threshold with validation
        mask_binary = (mask_np > threshold).astype(np.uint8)
        if np.sum(mask_binary) == 0:
            return None, mask_binary
        
        # Calculate minimum face size based on image dimensions
        image_size = max(mask_np.shape)
        min_face_size = (image_size * min_face_ratio) ** 2  # Square of min dimension
        
        # Find connected components
        try:
            labeled, num_features = ndimage.label(mask_binary)
            
            if num_features == 0:
                return None, mask_binary
                
        except Exception as e:
            return None, mask_binary
        
        # Find regions that could be faces based on relative size
        valid_regions = []
        for i in range(1, num_features + 1):
            region_mask = labeled == i
            region_size = np.sum(region_mask)
            
            # Skip regions smaller than minimum face size
            if region_size < min_face_size:
                continue
            
            # Get bounding box safely
            try:
                rows = np.any(region_mask, axis=1)
                cols = np.any(region_mask, axis=0)
                y_indices = np.where(rows)[0]
                x_indices = np.where(cols)[0]
                
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                    
                y_min, y_max = y_indices[[0, -1]]
                x_min, x_max = x_indices[[0, -1]]
                
                width = x_max - x_min
                height = y_max - y_min
                
                # Skip degenerate rectangles
                if width <= 0 or height <= 0:
                    continue
                
                # Store the region info with its area for sorting
                valid_regions.append((width * height, (x_min, y_min, x_max, y_max)))
                
            except Exception as e:
                continue
        
        if not valid_regions:
            return None, mask_binary
        
        # Sort by area and take the largest valid region
        valid_regions.sort(reverse=True)
        
        return valid_regions[0][1], mask_binary

    def apply_padding(self, bbox, image_shape, padding_factor):
        """Apply padding to bounding box with improved boundary handling"""
        x_min, y_min, x_max, y_max = bbox
        
        # Validate inputs
        if x_min >= x_max or y_min >= y_max:
            return bbox  # Return original bbox if invalid
            
        if image_shape[0] <= 0 or image_shape[1] <= 0:
            return bbox
        
        # Calculate center and size
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Validate dimensions
        if width <= 0 or height <= 0:
            return bbox
        
        # Calculate target size with padding
        target_width = max(1, int(width * padding_factor))  # Ensure minimum size of 1
        target_height = max(1, int(height * padding_factor))
        
        # Make square if dimensions are close (within 20%)
        size_ratio = min(target_width, target_height) / max(target_width, target_height)
        if size_ratio > 0.8:  # If dimensions are within 20% of each other
            target_size = max(target_width, target_height)
            target_width = target_height = target_size
        
        # Calculate new bounds with careful boundary handling
        new_x_min = max(0, center_x - target_width // 2)
        new_x_max = min(image_shape[1] - 1, center_x + (target_width - target_width // 2))
        new_y_min = max(0, center_y - target_height // 2)
        new_y_max = min(image_shape[0] - 1, center_y + (target_height - target_height // 2))
        
        # Handle edge cases where padding would push us outside the image
        # If we're at the edge, adjust to maintain the desired size if possible
        if new_x_min == 0:
            new_x_max = min(image_shape[1] - 1, new_x_min + target_width)
        if new_x_max == image_shape[1] - 1:
            new_x_min = max(0, new_x_max - target_width)
        if new_y_min == 0:
            new_y_max = min(image_shape[0] - 1, new_y_min + target_height)
        if new_y_max == image_shape[0] - 1:
            new_y_min = max(0, new_y_max - target_height)
        
        # Ensure dimensions are valid
        if new_x_max <= new_x_min or new_y_max <= new_y_min:
            return bbox
        
        # One last sanity check to ensure we're within bounds
        new_x_min = max(0, min(new_x_min, image_shape[1] - 2))
        new_x_max = max(new_x_min + 1, min(new_x_max, image_shape[1] - 1))
        new_y_min = max(0, min(new_y_min, image_shape[0] - 2))
        new_y_max = max(new_y_min + 1, min(new_y_max, image_shape[0] - 1))
        
        return int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)

    def crop_face(self, image, face_mask, padding_factor=1.2, threshold=0.5, min_face_ratio=0.01):
        """Crop face from image using face mask with robust error handling - adapted for ComfyUI's format"""
        try:
            # Ensure we have valid inputs
            if image is None or face_mask is None:
                # Return defaults that match expected ComfyUI dimensionality
                empty_mask = torch.zeros((1, image.shape[1], image.shape[2])) 
                return (image, 0, 0, image.shape[2], image.shape[1], empty_mask, empty_mask)
                
            # Ensure image is in [B, H, W, C] format as per ComfyUI
            if len(image.shape) != 4 or image.shape[3] not in [1, 3, 4]:
                # Try to convert if possible
                if len(image.shape) == 4 and image.shape[1] in [1, 3, 4]:
                    # Convert from [B, C, H, W] to [B, H, W, C]
                    image = image.permute(0, 2, 3, 1)
            
            # Ensure mask is in [B, H, W] format as per ComfyUI
            if len(face_mask.shape) != 3:
                # Try to convert if possible
                if len(face_mask.shape) == 4 and face_mask.shape[1] == 1:
                    # Convert from [B, C, H, W] to [B, H, W]
                    face_mask = face_mask[:, 0]
            
            # Find face bounding box and get binary mask
            bbox_result = self.find_main_face_bbox(face_mask, threshold, min_face_ratio)
            
            # If no face found, return original image and zeros
            if bbox_result[0] is None:
                # Return original image with zero mask
                empty_mask = torch.zeros((1, face_mask.shape[1], face_mask.shape[2]))
                return (image, 0, 0, image.shape[2], image.shape[1], empty_mask, empty_mask)
            
            bbox, binary_mask = bbox_result
            
            # Apply padding
            try:
                # For ComfyUI format, mask shape would be [B, H, W]
                # So image shape for padding is (H, W)
                image_shape = (face_mask.shape[1], face_mask.shape[2])
                
                x_min, y_min, x_max, y_max = self.apply_padding(
                    bbox, image_shape, padding_factor
                )
            except Exception as e:
                # Use original bbox without padding as fallback
                x_min, y_min, x_max, y_max = bbox
            
            # Double-check for valid crop dimensions
            if y_max <= y_min or x_max <= x_min:
                empty_mask = torch.zeros((1, face_mask.shape[1], face_mask.shape[2]))
                return (image, 0, 0, image.shape[2], image.shape[1], empty_mask, empty_mask)
                
            # Get proper dimensions for ComfyUI format [B, H, W, C]
            image_height = image.shape[1]
            image_width = image.shape[2]
            
            # Ensure crop region is within image bounds
            x_min = max(0, min(x_min, image_width - 1))
            y_min = max(0, min(y_min, image_height - 1))
            x_max = max(x_min + 1, min(x_max, image_width))
            y_max = max(y_min + 1, min(y_max, image_height))
            
            # Handle image tensor cropping for ComfyUI format [B, H, W, C]
            try:
                cropped = image[:, y_min:y_max, x_min:x_max, :]
                
                # Verify we have a valid crop
                if cropped.numel() == 0 or 0 in cropped.shape:
                    empty_mask = torch.zeros((1, face_mask.shape[1], face_mask.shape[2]))
                    return (image, 0, 0, image.shape[2], image.shape[1], empty_mask, empty_mask)
                    
            except Exception as e:
                empty_mask = torch.zeros((1, face_mask.shape[1], face_mask.shape[2]))
                return (image, 0, 0, image.shape[2], image.shape[1], empty_mask, empty_mask)
            
            # Create a mask for the face region - use ComfyUI format [B, H, W]
            try:
                final_crop_mask = torch.zeros((face_mask.shape[0], face_mask.shape[1], face_mask.shape[2]))
                final_crop_mask[:, y_min:y_max, x_min:x_max] = 1.0
            except Exception as e:
                # Create a basic fallback mask in ComfyUI format [B, H, W]
                final_crop_mask = torch.zeros((face_mask.shape[0], face_mask.shape[1], face_mask.shape[2]))
                final_crop_mask[:, y_min:y_max, x_min:x_max] = 1.0
            
            # Get final dimensions
            crop_width = x_max - x_min
            crop_height = y_max - y_min
            
            # Create custom mask from the original face_mask
            try:
                # Extract the face region from the original mask
                face_region = face_mask[:, y_min:y_max, x_min:x_max]
                
                # Make sure it has the right batch dimension
                if face_region.shape[0] != cropped.shape[0]:
                    face_region = face_region.expand(cropped.shape[0], -1, -1)
                
                # Normalize if needed
                if face_region.max() > 0:
                    face_region = face_region / face_region.max()
                
                # Use this as our custom mask
                custom_mask = face_region
                
            except Exception as e:
                # Create a fallback mask with the correct shape
                custom_mask = torch.ones((cropped.shape[0], cropped.shape[1], cropped.shape[2]))
            
            return (
                cropped,
                x_min,
                y_min,
                crop_width,
                crop_height,
                final_crop_mask,
                custom_mask
            )
            
        except Exception as e:
            # Return original image with default mask values
            empty_mask = torch.zeros((1, face_mask.shape[1], face_mask.shape[2]))
            return (image, 0, 0, image.shape[2], image.shape[1], empty_mask, empty_mask)

MAX_RESOLUTION = 8192
class Eden_ImageMaskComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "y": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "offset_x": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "offset_y": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱/face"
    
    def execute(self, destination, source, x, y, offset_x, offset_y, mask=None):
        # Diagnostic prints to help debug
        print(f"Source shape: {source.shape}")
        print(f"Destination shape: {destination.shape}")
        
        # Handle mask - create a default one if none provided
        if mask is None:
            print("No mask provided, creating default mask")
            mask = torch.ones((source.shape[0], source.shape[1], source.shape[2]))
        
        print(f"Original mask shape: {mask.shape}")
        
        # Normalize mask dimensions - handle various input formats
        # First, ensure mask has at least 3 dimensions [B, H, W]
        if len(mask.shape) == 2:  # [H, W]
            print("Expanding 2D mask to 3D")
            mask = mask.unsqueeze(0)  # Add batch dimension [1, H, W]
            
        # Now handle 3D masks [B, H, W] by adding channel dimension for broadcasting
        if len(mask.shape) == 3:  # [B, H, W] or [H, W, C]
            # Check if it's [H, W, C] format by seeing if third dimension is 1, 3, or 4 (typical channel sizes)
            if mask.shape[2] in [1, 3, 4] and mask.shape[0] != source.shape[0]:
                # This is likely [H, W, C] format, not [B, H, W]
                print("Detected [H, W, C] mask format, reshaping")
                mask = mask.permute(2, 0, 1)  # Convert to [C, H, W]
                mask = mask[0:1]  # Take first channel [1, H, W]
                mask = mask.unsqueeze(0)  # Ensure batch dim [1, 1, H, W]
            else:
                # This is [B, H, W] format, add channel dimension for broadcasting
                print("Expanding 3D mask by adding channel dimension")
                mask = mask.unsqueeze(-1)  # [B, H, W, 1]
        
        # At this point mask should be at least [B, H, W, 1]
        # Expand to 3 channels for RGB compositing if needed
        if mask.shape[-1] == 1:
            print("Repeating single channel mask to 3 channels")
            mask = mask.repeat(1, 1, 1, 3)  # Repeat last dimension to get [B, H, W, 3]
        
        print(f"Processed mask shape: {mask.shape}")
        
        # Resize mask to match source dimensions if needed
        if mask.shape[1:3] != source.shape[1:3]:
            print(f"Resizing mask from {mask.shape[1:3]} to {source.shape[1:3]}")
            try:
                # Convert to channels-first for F.interpolate
                mask_for_resize = mask.permute(0, 3, 1, 2)  # [B, C, H, W]
                mask_resized = F.interpolate(
                    mask_for_resize, 
                    size=(source.shape[1], source.shape[2]), 
                    mode='bicubic',
                    align_corners=False
                )
                # Convert back to channels-last
                mask = mask_resized.permute(0, 2, 3, 1)  # [B, H, W, C]
                print(f"Resized mask shape: {mask.shape}")
            except Exception as e:
                print(f"Error resizing mask: {e}")
                # Create a simple fallback mask matching source dimensions
                mask = torch.ones((source.shape[0], source.shape[1], source.shape[2], 3))
        
        # Handle batch dimension mismatches
        if mask.shape[0] > source.shape[0]:
            print(f"Reducing mask batch dim from {mask.shape[0]} to {source.shape[0]}")
            mask = mask[:source.shape[0]]
        elif mask.shape[0] < source.shape[0]:
            print(f"Expanding mask batch dim from {mask.shape[0]} to {source.shape[0]}")
            # Repeat the last mask in batch to match source batch size
            mask = torch.cat((mask, mask[-1:].repeat((source.shape[0]-mask.shape[0], 1, 1, 1))), dim=0)
        
        # Normalize destination batch size to match source
        if destination.shape[0] > source.shape[0]:
            print(f"Reducing destination batch dim from {destination.shape[0]} to {source.shape[0]}")
            destination = destination[:source.shape[0]]
        elif destination.shape[0] < source.shape[0]:
            print(f"Expanding destination batch dim from {destination.shape[0]} to {source.shape[0]}")
            destination = torch.cat((destination, destination[-1:].repeat((source.shape[0]-destination.shape[0], 1, 1, 1))), dim=0)
        
        # Handle x,y positioning parameters
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        
        # Extend x,y lists to match batch size if needed
        if len(x) < destination.shape[0]:
            x = x + [x[-1]] * (destination.shape[0] - len(x))
        if len(y) < destination.shape[0]:
            y = y + [y[-1]] * (destination.shape[0] - len(y))
        
        # Apply offsets
        x = [i + offset_x for i in x]
        y = [i + offset_y for i in y]
        
        output = []
        for i in range(destination.shape[0]):
            d = destination[i].clone()
            s = source[i]
            m = mask[i]
            
            # Safety check for shapes
            print(f"Batch {i}: d={d.shape}, s={s.shape}, m={m.shape}")
            
            # Get composite region dimensions to ensure safe indexing
            h_src, w_src = s.shape[0], s.shape[1]
            h_dst, w_dst = d.shape[0], d.shape[1]
            
            # Calculate valid region bounds
            x_start = max(0, x[i])
            y_start = max(0, y[i])
            x_end = min(w_dst, x[i] + w_src)
            y_end = min(h_dst, y[i] + h_src)
            
            # Skip if no valid region
            if x_end <= x_start or y_end <= y_start:
                print(f"No valid overlap region for batch {i}, skipping composite")
                output.append(d)
                continue
            
            # Calculate how much of source to use
            s_x_start = 0 if x[i] >= 0 else -x[i]
            s_y_start = 0 if y[i] >= 0 else -y[i]
            s_x_end = w_src - (x[i] + w_src - x_end if x[i] + w_src > x_end else 0)
            s_y_end = h_src - (y[i] + h_src - y_end if y[i] + h_src > y_end else 0)
            
            # Safety checks for valid slices
            if s_x_end <= s_x_start or s_y_end <= s_y_start:
                print(f"Invalid source region for batch {i}, skipping composite")
                output.append(d)
                continue
            
            try:
                # Extract slices from source and mask
                s_region = s[s_y_start:s_y_end, s_x_start:s_x_end, :]
                m_region = m[s_y_start:s_y_end, s_x_start:s_x_end, :]
                
                # Extract destination region
                d_region = d[y_start:y_end, x_start:x_end, :]
                
                # Ensure dimensions match
                if s_region.shape != m_region.shape[:3]:
                    print(f"Shape mismatch between source region {s_region.shape} and mask region {m_region.shape}")
                    # Resize mask to match source if needed
                    m_region = F.interpolate(
                        m_region.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
                        size=s_region.shape[:2],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # Back to [H, W, C]
                
                # Perform composite
                d[y_start:y_end, x_start:x_end, :] = (
                    s_region * m_region + 
                    d_region * (1 - m_region)
                )
                
            except Exception as e:
                print(f"Error during composite for batch {i}: {e}")
                # Continue to next batch without modifying this one
            
            output.append(d)
        
        output = torch.stack(output)
        print(f"Final output shape: {output.shape}")
        return (output,)

def round_to_nearest_multiple(number, multiple):
    return int(multiple * round(number / multiple))

def get_centre_crop(img, aspect_ratio):
    h, w = np.array(img).shape[:2]
    if w/h > aspect_ratio:
        # crop width:
        new_w = int(h * aspect_ratio)
        left = (w - new_w) // 2
        right = (w + new_w) // 2
        crop = img[:, left:right]
    else:
        # crop height:
        new_h = int(w / aspect_ratio)
        top = (h - new_h) // 2
        bottom = (h + new_h) // 2
        crop = img[top:bottom, :]
    return crop

def get_uniformly_sized_crops(imgs, target_n_pixels=2048**2):
    """
    Given a list of images:
        - extract the best possible centre crop of same aspect ratio for all images
        - rescale these crops to have ~target_n_pixels
        - return resized images
    """

    assert len(imgs) > 1
    imgs = [np.array(img) for img in imgs]
    
    # Get center crops at same aspect ratio
    aspect_ratios = [img.shape[1] / img.shape[0] for img in imgs]
    final_aspect_ratio = np.mean(aspect_ratios)
    crops = [get_centre_crop(img, final_aspect_ratio) for img in imgs]

    # Compute final w,h using final_aspect_ratio and target_n_pixels:
    final_h = np.sqrt(target_n_pixels / final_aspect_ratio)
    final_w = final_h * final_aspect_ratio
    final_h = round_to_nearest_multiple(final_h, 8)
    final_w = round_to_nearest_multiple(final_w, 8)

    # Resize images
    resized_imgs = [cv2.resize(crop, (final_w, final_h), cv2.INTER_CUBIC) for crop in crops]
    #resized_imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in resized_imgs]
    
    return resized_imgs

class LoadRandomImage:
    def __init__(self):
        self.img_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "folder": ("STRING", {"default": "."}),
                    "n_images": ("INT", {"default": 1, "min": -1, "max": 100}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 100000}),
                    "sort": ("BOOLEAN", {"default": False}),
                    "loop_sequence": ("BOOLEAN", {"default": False}),
                }
        }

    CATEGORY = "Eden 🌱/general"
    RETURN_TYPES = ("IMAGE", any_typ, any_typ, "STRING")
    RETURN_NAMES = ("Image(s)", "paths", "filenames", "filenames[0]_str")
    FUNCTION = "load"

    def load(self, folder, n_images, seed, sort, loop_sequence):
        image_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        image_paths = [f for f in image_paths if os.path.isfile(f)]
        # Filter using file extensions
        image_paths = [f for f in image_paths if any([f.endswith(ext) for ext in self.img_extensions])]
        valid_image_paths = []
        
        for f in image_paths:
            if imghdr.what(f):
                valid_image_paths.append(f)
            else:
                try:
                    img = Image.open(f)
                    img.verify()  # Ensure it's a valid image
                    valid_image_paths.append(f)
                except Exception as e:
                    print(f"Skipping invalid image: {f} - {str(e)}")

        # Special case: sort=True and n_images=1 → use seed as index
        if sort and n_images == 1:
            valid_image_paths = sorted(valid_image_paths)
            if valid_image_paths:
                idx = seed % len(valid_image_paths)
                valid_image_paths = [valid_image_paths[idx]]
        else:
            random.seed(seed)
            random.shuffle(valid_image_paths)
            if n_images > 0:
                valid_image_paths = valid_image_paths[:n_images]
            if sort:
                valid_image_paths = sorted(valid_image_paths)

        imgs, paths, filenames = [], [], []
        for image_path in valid_image_paths:
            try:
                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img)
            except Exception as e:
                print(f"Error during EXIF transpose for {image_path}: {str(e)}")

            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))

            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            imgs.append(image)
            paths.append(image_path)
            filenames.append(os.path.basename(image_path))

        if loop_sequence and len(imgs) > 1:
            imgs.append(imgs[0])
            paths.append(paths[0])
            filenames.append(filenames[0])

        if len(imgs) > 1:
            imgs = get_uniformly_sized_crops(imgs, target_n_pixels=1024**2)
            imgs = [torch.from_numpy(img)[None,] for img in imgs]
            output_image = torch.cat(imgs, dim=0)
        else:
            output_image = torch.from_numpy(imgs[0])[None,]

        return (output_image, paths, filenames, str(filenames[0]))


class ImageFolderIterator:
    def __init__(self):
        self.img_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG"]
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": "."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "sort": ("BOOLEAN", {"default": True}),
            }
        }
    
    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "load_image"
    
    def get_image_paths(self, folder, sort):
        # List all files in the folder
        image_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        image_paths = [f for f in image_paths if os.path.isfile(f)]
        # Filter using file extensions
        image_paths = [f for f in image_paths if any([f.endswith(ext) for ext in self.img_extensions])]
        valid_image_paths = []
        
        # Validate images
        for f in image_paths:
            if imghdr.what(f):
                valid_image_paths.append(f)
            else:
                try:
                    img = Image.open(f)
                    img.verify()
                    valid_image_paths.append(f)
                except Exception as e:
                    print(f"Skipping invalid image: {f} - {str(e)}")
        
        # Sort if requested
        if sort:
            valid_image_paths = sorted(valid_image_paths)
            
        return valid_image_paths
    
    def load_image(self, folder, index, sort):
        valid_image_paths = self.get_image_paths(folder, sort)
        
        if not valid_image_paths:
            raise ValueError(f"No valid images found in folder: {folder}")
        
        # Wrap around if index exceeds number of images
        actual_index = index % len(valid_image_paths)
        image_path = valid_image_paths[actual_index]
        
        try:
            # Load and process image
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)  # Correct orientation based on EXIF
            
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            
            # Add batch dimension
            output_image = torch.from_numpy(image)[None,]
            
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(image_path))[0]
            
            return (output_image, filename)
            
        except Exception as e:
            raise RuntimeError(f"Error processing image {image_path}: {str(e)}")
        
        
class LoadImagesByFilename:
    def __init__(self):
        self.img_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "filename": ("COMBO", {"default": []}),
                    "max_num_images": ("INT", {"default": None, "min": 0, "max": sys.maxsize}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 100000}),
                    "sort": ("BOOLEAN", {"default": False}),
                    "loop_sequence": ("BOOLEAN", {"default": False}),
                }
        }

    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"

    def load_image(self, filename: list, max_num_images, seed, sort, loop_sequence):
        """
        Loads images from a list of filenames as input: `filename`
        """
        files = filename

        random.seed(seed)
        random.shuffle(files)

        if max_num_images == 0:
            max_num_images = None
        image_paths = files[:max_num_images]

        if sort:
            image_paths = sorted(image_paths)

        imgs = [Image.open(image_path) for image_path in image_paths]
        output_images = []
        for img in imgs:
            img = ImageOps.exif_transpose(img)
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            output_images.append(image)

        if loop_sequence:
            # Make sure the last image is the same as the first image:
            output_images.append(output_images[0])

        if len(output_images) > 1:
            output_images = get_uniformly_sized_crops(output_images, target_n_pixels=1024**2)
            output_images = [torch.from_numpy(output_image)[None,] for output_image in output_images]
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = torch.from_numpy(output_images[0])[None,]
        return (output_image,)





class GetRandomFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "folder": ("STRING", {"default": "."}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 100000}),
                }
        }

    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_path"

    def get_path(self, folder, seed):
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
        files = [f for f in files if os.path.isfile(f)]

        random.seed(seed)
        path = random.choice(files)
        return (path,)



class IMG_resolution_multiple_of:
    def __init__(self):
        self.ci = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "multiple_of": ("INT", {"default": 8, "min": 2, "max": 264}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pad"
    CATEGORY = "Eden 🌱/Image"

    def pad(self, image, multiple_of):
        bs, h, w, c = image.shape
        # Crop the image to the nearest multiple of 8:
        h = h - h % multiple_of
        w = w - w % multiple_of
        image = image[:, :h, :w, :]
        return (image,)

class IMG_padder:
    def __init__(self):
        self.ci = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pad_fraction": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.01}),
                "pad_location": (["bottom", "top", "left", "right"], ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pad"
    CATEGORY = "Eden 🌱/Image"

    def pad(self, image, pad_fraction, pad_location="bottom"):

        bs, h, w, c = image.shape

        color_mean_w = 4 # pixels

        # add a black border:
        if pad_location == "bottom":
            border = torch.ones((bs, int(h * pad_fraction), w, c))
            # Match the mean color at the bottom edge:
            border = border * image[:, -color_mean_w:, :, :].mean()
            image = torch.cat((image, border), dim=1)
        elif pad_location == "top":
            border = torch.ones((bs, int(h * pad_fraction), w, c))
            # Match the mean color at the top edge:
            border = border * image[:, :color_mean_w, :, :].mean()
            image = torch.cat((border, image), dim=1)
        elif pad_location == "left":
            border = torch.ones((bs, h, int(w * pad_fraction), c))
            # Match the mean color at the left edge:
            border = border * image[:, :, :color_mean_w, :].mean()
            image = torch.cat((border, image), dim=2)
        elif pad_location == "right":
            border = torch.ones((bs, h, int(w * pad_fraction), c))
            # Match the mean color at the right edge:
            border = border * image[:, :, -color_mean_w:, :].mean()
            image = torch.cat((image, border), dim=2)

        return (image,)



    
class IMG_blender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image1_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "Eden 🌱/Image"

    def blend(self, image1, image2, image1_weight = 0.5):

        bs1, h1, w1, c1 = image1.shape
        bs2, h2, w2, c2 = image2.shape

        if bs1 != bs2:
            raise ValueError("Images must have the same batch size for blending!")

        if h1 != h2 or w1 != w2:
            # simply crop the larger image to the size of the smaller one:
            h = min(h1, h2)
            w = min(w1, w2)
            image1 = image1[:, :h, :w, :]
            image2 = image2[:, :h, :w, :]

        blended_image = image1 * image1_weight + image2 * (1 - image1_weight)
        
        return (blended_image,)


    
class IMG_unpadder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "unpad_fraction": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.01}),
                "unpad_location": (["bottom", "top", "left", "right"], ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "unpad"
    CATEGORY = "Eden 🌱/Image"

    def unpad(self, image, unpad_fraction, unpad_location = "bottom"):

        bs, h, w, c = image.shape

        if unpad_location == "bottom":
            image = image[:, :int(h * (1 - unpad_fraction)), :, :]
        elif unpad_location == "top":
            image = image[:, int(h * unpad_fraction):, :, :]
        elif unpad_location == "left":
            image = image[:, :, int(w * unpad_fraction):, :]
        elif unpad_location == "right":
            image = image[:, :, :int(w * (1 - unpad_fraction)), :]

        # always make sure width and height are divisible by 4:
        h = image.shape[1]
        w = image.shape[2]
        h = h - h % 4
        w = w - w % 4
        image = image[:, :h, :w, :]
        
        return (image,)
    

class IMG_scaler:
    """
    A class to apply mathematical operations to an image.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                 "math_string": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_math"
    CATEGORY = "Eden 🌱/Image"

    def apply_math(self, image, math_string):
        '''
        Apply a mathematical operation to the image.
        The math_string is applied to each pixel value.
        '''

        # Ensure the input is a PyTorch tensor
        if not isinstance(image, torch.Tensor):
            raise TypeError("The image must be a PyTorch tensor")

        input_device = image.device
        input_dtype  = image.dtype

        # Normalize image if needed
        if image.max() > 1:
            image = image.float() / 255.0

        # Define the mathematical function securely
        def safe_eval(expr, x):
            allowed_functions = {"sin": np.sin, "cos": np.cos, "exp": np.exp}  # Extend this as needed
            return eval(expr, {"__builtins__": None}, allowed_functions)

        # Create a vectorized version of the safe_eval function
        vectorized_func = np.vectorize(lambda x: safe_eval(math_string, x))

        # Apply the function to the numpy version of the image
        transformed_image = torch.from_numpy(vectorized_func(image.cpu().numpy()))

        # Clip and rescale if necessary
        transformed_image = torch.clamp(transformed_image, 0, 1)
        if input_dtype == torch.uint8:
            transformed_image = (transformed_image * 255).to(torch.uint8)

        # Convert back to original device and dtype
        output_image = transformed_image.to(input_device, dtype=input_dtype)

        return (output_image,)


def to_grayscale(images, keep_dims=True, alpha_channel_convert_to=None):
    """
    Convert a batch of RGB or RGBA images to grayscale.
    
    Args:
        images (torch.Tensor): Input tensor of shape (batch_size, height, width, channels)
                                with channels = 3 for RGB images or 4 for RGBA images.
        keep_dims (bool): If True, maintain the original number of channels.
        alpha_channel_convert_to (tuple or None): RGB values to set for transparent pixels.
    
    Returns:
        torch.Tensor: Grayscale images of shape (batch_size, height, width, channels).
    """
    if images.shape[-1] not in [3, 4]:
        raise ValueError("Input images must have 3 (RGB) or 4 (RGBA) channels.")

    # Define the weights for the RGB channels to convert to grayscale.
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=images.device)

    # Separate the alpha channel if present
    if images.shape[-1] == 4:
        rgb = images[..., :3]
        alpha = images[..., 3:]
    else:
        rgb = images
        alpha = None

    # Permute the dimensions to (batch_size, channels, height, width)
    rgb_permuted = rgb.permute(0, 3, 1, 2)

    # Perform the weighted sum along the channel dimension (dim=1)
    grayscale_images = torch.tensordot(rgb_permuted, weights, dims=([1], [0]))

    # Add an extra dimension for the channel at the end.
    grayscale_images = grayscale_images.unsqueeze(-1)

    if keep_dims:
        # Repeat the grayscale image to match the original dimensions.
        grayscale_images = grayscale_images.repeat(1, 1, 1, 3)
        
        if alpha is not None:
            # If alpha_channel_convert_to is provided, blend the grayscale with the specified color
            if alpha_channel_convert_to is not None:
                grayscale_images = grayscale_images * alpha + alpha_channel_convert_to * (1 - alpha)

    return grayscale_images


class ConvertToGrayscale:
    def __init__(self):
        self.ci = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_channel_convert_to": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.01})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_to_grayscale"
    CATEGORY = "Eden 🌱/Image"

    def convert_to_grayscale(self, image, alpha_channel_convert_to):
        # Input is a torch tensor with shape (bs, h, w, c)
        bs, h, w, c = image.shape
        if c == 1:
            return (image,)
        elif c in [3, 4]:
            # Convert the image to grayscale
            image = to_grayscale(image, keep_dims=True, alpha_channel_convert_to=alpha_channel_convert_to)
        else:
            raise ValueError(f"Input image must have 1, 3, or 4 channels, but got {c} channels. Image shape = {image.shape}")
        return (image,)


class AspectPadImageForOutpainting:
    def __init__(self):
        pass
    
    """
    A node to calculate args for default comfy node 'Pad Image For Outpainting'
    """
    ASPECT_RATIO_MAP = {
        "1-1_square_1024x1024": (1024, 1024),
        "4-3_landscape_1152x896": (1152, 896),
        "3-2_landscape_1216x832": (1216, 832),
        "16-9_landscape_1344x768": (1344, 768),
        "21-9_landscape_1536x640": (1536, 640),
        "3-4_portrait_896x1152": (896, 1152),
        "2-3_portrait_832x1216": (832, 1216),
        "9-16_portrait_768x1344": (768, 1344),
        "9-21_portrait_640x1536": (640, 1536),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (list(s.ASPECT_RATIO_MAP.keys()), {"default": "16-9_landscape_1344x768"}),
                "justification": (["top-left", "center", "bottom-right"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE","LEFT","TOP","RIGHT","BOTTOM")
    FUNCTION = "fit_and_calculate_padding"
    CATEGORY = "Eden 🌱"

    def fit_and_calculate_padding(self, image, aspect_ratio, justification):
        bs, h, w, c = image.shape

        # Get the canvas dimensions from the aspect ratio map
        canvas_width, canvas_height = self.ASPECT_RATIO_MAP[aspect_ratio]

        # Calculate the aspect ratios
        image_aspect_ratio = w / h
        canvas_aspect_ratio = canvas_width / canvas_height

        # Determine the new dimensions
        if image_aspect_ratio > canvas_aspect_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / image_aspect_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * image_aspect_ratio)

        # Resize the image
        resized_image = torch.nn.functional.interpolate(image.permute(0, 3, 1, 2), size=(new_height, new_width), mode='bicubic', align_corners=False)
        resized_image = resized_image.permute(0, 2, 3, 1)

        # Calculate padding
        if justification == "center":
            left = (canvas_width - new_width) // 2
            right = canvas_width - new_width - left
            top = (canvas_height - new_height) // 2
            bottom = canvas_height - new_height - top
        elif justification == "top-left":
            left = 0
            right = canvas_width - new_width
            top = 0
            bottom = canvas_height - new_height
        elif justification == "bottom-right":
            left = canvas_width - new_width
            right = 0
            top = canvas_height - new_height
            bottom = 0

        return (resized_image, left, top, right, bottom)
    

class Extend_Sequence:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  # Input is a stack of images
                "target_n_frames": ("INT", {"default": 24, "min": 1, "step": 1, "max": sys.maxsize}),  # Desired output number of frames
                "mode": (["wrap_around", "ping_pong"], ),  # Various modes for handling the sequence
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_sequence"
    CATEGORY = "Eden 🌱/Image"

    def process_sequence(self, images, target_n_frames, mode="wrap_around"):
        n_frames = images.shape[0]

        if mode == "wrap_around":
            # Wrap the sequence around to get target_n_frames
            extended_images = self._wrap_around(images, target_n_frames)
        elif mode == "ping_pong":
            # Repeat and reverse the sequence to create a ping-pong effect
            extended_images = self._ping_pong(images, target_n_frames)

        return (extended_images,)

    def _wrap_around(self, images, target_n_frames):
        """Wrap around the input images to match the target number of frames."""
        n_frames = images.shape[0]
        # Use torch indexing to repeat images without new allocations
        indices = torch.arange(target_n_frames) % n_frames
        return images[indices]

    def _ping_pong(self, images, target_n_frames):
        """Create a ping-pong effect by repeating and reversing frames."""
        n_frames = images.shape[0]
        indices = torch.arange(target_n_frames) % (2 * n_frames)
        indices = torch.where(indices >= n_frames, 2 * n_frames - indices - 1, indices)
        return images[indices]