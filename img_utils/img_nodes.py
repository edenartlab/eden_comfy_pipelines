import torch
from PIL import Image
import os, time
import torch
import cv2
import numpy as np
import random
import gc
import torch
import imghdr

###########################################################################

# Import comfyUI modules:
from cli_args import args
import folder_paths

###########################################################################

from torch.cuda.amp import autocast
import psutil

def print_available_memory():
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / 1024 / 1024 / 1024:.2f} GB")

import torchvision.transforms.functional as T
import comfy.utils

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
                    {"width":  ("INT", {"default": 512}),
                     "height":  ("INT", {"default": 512}),
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

import numpy as np
import torch.nn.functional as F
import torch

def gaussian_kernel_2d(sigma, size=0):
    size = int(2 * sigma) if size == 0 else size
    kernel_size = size // 2 * 2  # Ensure even size
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, -1, 1) * kernel.view(1, 1, 1, -1)

from sklearn.cluster import KMeans
from .img_utils import lab_to_rgb, rgb_to_lab
import numpy as np

class MaskFromRGB_KMeans:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "n_color_clusters": ("INT", {"default": 6, "min": 2, "max": 10}),
                "clustering_resolution": ("INT", {"default": 256, "min": 32, "max": 1024}),
                "feathering_fraction": ("FLOAT", { "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01 }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("1","2","3","4","5","6","7","8",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱"

    @torch.no_grad()
    def execute(self, image, n_color_clusters, clustering_resolution, feathering_fraction):
        # Assuming you have your batch of PyTorch image tensors called 'image_batch'
        # Shape of image_batch: [n, h, w, 3]
        image = image.cuda()
        lab_images = torch.stack([rgb_to_lab(img) for img in image])
        n, h, w, _ = lab_images.shape

        # bring channel dim to second position:
        lab_images = lab_images.permute(0, 3, 1, 2)

        # Make sure to maintain aspect ratio:
        h_target, w_target = clustering_resolution, int(clustering_resolution * w / h)
        lab_images = F.interpolate(lab_images, size=[h_target, w_target], mode='bicubic', align_corners=False)
        # bring channel dim back to last position:
        lab_images = lab_images.permute(0, 2, 3, 1)

        # Reshape images to [n*w*h, 3] for k-means clustering
        n, h, w, _ = lab_images.shape
        lab_images_reshaped = lab_images.view(n*w*h, 3)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_color_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(lab_images_reshaped.cpu().numpy())
        
        # Calculate average luminance for each cluster
        cluster_centers = kmeans.cluster_centers_
        cluster_luminance = cluster_centers[:, 0]  # L channel in LAB color space
        
        # Sort cluster indices based on luminance:
        sorted_indices = np.argsort(cluster_luminance)
        index_map = {old: new for new, old in enumerate(sorted_indices)}
        
        # Map the cluster labels to new sorted indices
        sorted_cluster_labels = np.vectorize(index_map.get)(cluster_labels)
        cluster_labels = torch.from_numpy(sorted_cluster_labels).to("cpu").view(n, h, w)

        # Transform the cluster_labels into masks:
        masks = torch.zeros(n, 8, h, w, device=image.device)

        for i in range(n):
            for j in range(n_color_clusters):
                masks[i, j] = (cluster_labels[i] == j).float()

        if feathering_fraction > 0:
            masks = masks.to("cuda")
            n_imgs, n_colors, h, w = masks.shape
            batch_size = n_imgs * n_colors
            masks = masks.view(batch_size, h, w)

            feathering = int(feathering_fraction * (w+h)/2)
            feathering = feathering // 2 * 2

            kernel = gaussian_kernel_2d(feathering).to(masks.device)
            
            # Calculate padding size
            pad_size = kernel.shape[2] // 2

            print("Feathering masks...")
            # Apply convolution for feathering
            masks_feathered = torch.zeros_like(masks)
            for i in range(masks.shape[0]):
                mask_padded = masks[i]
                mask_padded = mask_padded.unsqueeze(0).unsqueeze(0)  # Add batch dimension
                
                # Apply reflection padding
                mask_padded = F.pad(mask_padded, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
                
                # Apply convolution
                mask_feathered = F.conv2d(mask_padded, kernel, padding=0)
                
                # Ensure the output size matches the input size exactly
                mask_feathered = F.interpolate(mask_feathered, size=(h, w), mode='bilinear', align_corners=False)
                
                masks_feathered[i] = mask_feathered.squeeze()
            
            masks = masks_feathered.view(n_imgs, n_colors, h, w).to("cpu")

        # Upscale masks to original resolution:
        masks = F.interpolate(masks, size=(image.shape[1], image.shape[2]), mode='bicubic', align_corners=False).to("cpu")

        return masks[:, 0], masks[:, 1], masks[:, 2], masks[:, 3], masks[:, 4], masks[:, 5], masks[:, 6], masks[:, 7]



from PIL.PngImagePlugin import PngInfo
import json

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

from PIL import Image, ImageOps, ImageSequence
class LoadRandomImage:
    def __init__(self):
        self.img_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

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

    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"

    def load_image(self, folder, n_images, seed, sort, loop_sequence):
        image_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        image_paths = [f for f in image_paths if os.path.isfile(f)]
        # filter using file extensions:
        image_paths = [f for f in image_paths if any([f.endswith(ext) for ext in self.img_extensions])]
        # filter using image headers:
        image_paths = [f for f in image_paths if imghdr.what(f)]

        random.seed(seed)
        random.shuffle(image_paths)

        if n_images > 0:
            image_paths = image_paths[:n_images]

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


class LoadImagesByFilename:
    def __init__(self):
        self.img_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "filename": ("COMBO", {"default": []}),
                    "max_num_images": ("INT", {"default": None}),
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
                "target_n_frames": ("INT", {"default": 24, "min": 1, "step": 1}),  # Desired output number of frames
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