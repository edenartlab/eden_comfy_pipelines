import torch
from PIL import Image
import sys, os, time
import torch
import cv2
import numpy as np
import random
import gc
import torch
from torchvision import transforms
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

class MaskFromRGB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "threshold_r": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_g": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_b": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "feathering": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 500.0, "step": 0.1 }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("red","green","blue","cyan","magenta","yellow","black","white",)
    FUNCTION = "execute"
    CATEGORY = "Eden 🌱"

    @torch.no_grad()
    def execute(self, image, threshold_r, threshold_g, threshold_b, feathering):

        # Thresholding
        red = (image[..., 0] >= 1 - threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] < threshold_b)
        green = (image[..., 0] < threshold_r) & (image[..., 1] >= 1 - threshold_g) & (image[..., 2] < threshold_b)
        blue = (image[..., 0] < threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] >= 1 - threshold_b)

        cyan = (image[..., 0] < threshold_r) & (image[..., 1] >= 1 - threshold_g) & (image[..., 2] >= 1 - threshold_b)
        magenta = (image[..., 0] >= 1 - threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] > 1 - threshold_b)
        yellow = (image[..., 0] >= 1 - threshold_r) & (image[..., 1] >= 1 - threshold_g) & (image[..., 2] < threshold_b)

        black = (image[..., 0] <= threshold_r) & (image[..., 1] <= threshold_g) & (image[..., 2] <= threshold_b)
        white = (image[..., 0] >= 1 - threshold_r) & (image[..., 1] >= 1 - threshold_g) & (image[..., 2] >= 1 - threshold_b)

        # Combine masks
        masks = torch.stack([red, green, blue, cyan, magenta, yellow, black, white], dim=1).float()

        if feathering > 0:
            masks = masks.to("cuda")
            n_imgs, n_colors, h, w = masks.shape
            batch_size = n_imgs * n_colors
            masks = masks.view(batch_size, h, w)

            kernel = gaussian_kernel_2d(feathering).to(masks.device)

            print("Feathering masks...")
            # Apply convolution for feathering
            masks_feathered = torch.zeros_like(masks)
            for i in range(masks.shape[0]):
                mask_padded = masks[i]
                mask_padded = mask_padded.unsqueeze(0).unsqueeze(0)  # Add batch dimension
                mask_feathered = F.conv2d(mask_padded, kernel, padding='same')
                masks_feathered[i] = mask_feathered.squeeze()
            
            masks = masks_feathered.view(n_imgs, n_colors, h, w).to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        # convert masks to float16 to save memory:
        masks = masks.half()

        return masks[:, 0], masks[:, 1], masks[:, 2], masks[:, 3], masks[:, 4], masks[:, 5], masks[:, 6], masks[:, 7]

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
        masks = torch.zeros(n, 8, h, w)

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

            print("Feathering masks...")
            # Apply convolution for feathering
            masks_feathered = torch.zeros_like(masks)
            for i in range(masks.shape[0]):
                mask_padded = masks[i]
                mask_padded = mask_padded.unsqueeze(0).unsqueeze(0)  # Add batch dimension
                mask_feathered = F.conv2d(mask_padded, kernel, padding='same')
                masks_feathered[i] = mask_feathered.squeeze()
            
            masks = masks_feathered.view(n_imgs, n_colors, h, w).to("cpu")

        # Upscale masks to original resolution:
        masks = F.interpolate(masks, size=(image.shape[1], image.shape[2]), mode='bicubic', align_corners=False)

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
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
        files = [f for f in files if os.path.isfile(f)]
        # filter using file extensions:
        files = [f for f in files if any([f.endswith(ext) for ext in self.img_extensions])]
        # filter using image headers:
        files = [f for f in files if imghdr.what(f)]

        random.seed(seed)
        random.shuffle(files)

        if sort:
            files = sorted(files)

        print(f"Sorted files:")
        for f in files:
            print(f)

        if n_images > 0:
            image_paths = files[:n_images]

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
                    "filename": ("LIST",),
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



import torch
import cv2
import numpy as np

class HIST_matcher_depracted:
    def __init__(self):
        self.ci = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_image": ("IMAGE",),
                "dst_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "hist_match"
    CATEGORY = "Eden 🌱/Image"

    def hist_match(self, src_image, dst_images):
        # bs, h, w, c = src_image.shape

        # Convert images to numpy arrays
        src_image_np  = 255. * src_image.cpu().numpy()
        dst_images_np = 255. * dst_images.cpu().numpy()

        # clip to 0-255:
        src_image_np  = np.clip(src_image_np, 0, 255).astype(np.uint8)
        dst_images_np = np.clip(dst_images_np, 0, 255).astype(np.uint8)

        # Convert images to YCrCb color space
        input_img_ycrcb   = cv2.cvtColor(src_image_np[0], cv2.COLOR_BGR2YCrCb)
        output_imgs_ycrcb = [cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) for img in dst_images_np]

        # Compute the histogram of the input image
        hist_input = cv2.calcHist([input_img_ycrcb], [0], None, [256], [0, 256])

        # Compute the average histogram of the output images
        hist_output_avg = np.mean([cv2.calcHist([img], [0], None, [256], [0, 256]) for img in output_imgs_ycrcb], axis=0)   

        # Create a lookup table to map the average output histogram to the input histogram
        cumulative_input = np.cumsum(hist_input) / sum(hist_input)
        cumulative_output = np.cumsum(hist_output_avg) / sum(hist_output_avg)

        lookup_table = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            idx = np.abs(cumulative_input[i] - cumulative_output).argmin()
            lookup_table[i] = idx

        # Visualize the lookup table:
        plt.figure()
        plt.plot(lookup_table)
        plt.savefig("lookup_table.png")
        
        # Apply the lookup table to the Y channel of each output image
        adjusted_imgs = []

        plt.figure()
        for img in output_imgs_ycrcb:
            img_adjusted = cv2.LUT(img[:,:,0], lookup_table)
            img_adjusted = cv2.merge([img_adjusted, img[:,:,1], img[:,:,2]])

            adjusted_hist = cv2.calcHist([img_adjusted], [0], None, [256], [0, 256])
            plt.plot(adjusted_hist, label="Adjusted")

            # Convert back to BGR color space and add to the list of adjusted images
            img_adjusted_bgr = cv2.cvtColor(img_adjusted, cv2.COLOR_YCrCb2BGR)
            torch_img = torch.tensor(img_adjusted_bgr).float() / 255.0
            adjusted_imgs.append(torch_img)

        plt.legend()
        plt.savefig("adjusted_histograms.png")

        output_tensors = torch.stack(adjusted_imgs, dim=0)

        return (output_tensors,)




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


def to_grayscale(images, keep_dims=True):
    """
    Convert a batch of RGB images to grayscale.
    
    Args:
        images (torch.Tensor): Input tensor of shape (batch_size, height, width, channels)
                                with channels = 3 for RGB images.
    
    Returns:
        torch.Tensor: Grayscale images of shape (batch_size, height, width, 1) or (batch_size, height, width, 3) if keep_dims=True.
    """
    if images.shape[-1] != 3:
        raise ValueError("Input images must have 3 channels (RGB).")

    # Define the weights for the RGB channels to convert to grayscale.
    # These are the standard weights used in the ITU-R BT.601 standard.
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=images.device)

    # Permute the dimensions to (batch_size, channels, height, width)
    # for easier matrix multiplication.
    images_permuted = images.permute(0, 3, 1, 2)

    # Perform the weighted sum along the channel dimension (dim=1)
    grayscale_images = torch.tensordot(images_permuted, weights, dims=([1], [0]))

    # Add an extra dimension for the channel at the end.
    grayscale_images = grayscale_images.unsqueeze(-1)

    if keep_dims:
        # Permute the dimensions back to (batch_size, height, width, 1)
        #grayscale_images = grayscale_images.permute(0, 2, 3, 1)
        # Repeat the grayscale image 3 times to match the original dimensions.
        grayscale_images = grayscale_images.repeat(1, 1, 1, 3)
    

    return grayscale_images


class ConvertToGrayscale:
    def __init__(self):
        self.ci = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_to_grayscale"
    CATEGORY = "Eden 🌱/Image"

    def convert_to_grayscale(self, image):
        # Input is a torch tensor with shape (bs, c, h, w)
        bs, h, w, c = image.shape
        if c == 1:
            pass
        elif c == 3:
            # Convert the image to grayscale
            image = to_grayscale(image)
        else:
            raise ValueError(f"Input image must have 1 or 3 channels, but got {c} channels. Image shape = {image.shape}")
        return (image,)

class AspectPadImageForOutpainting:
    """
    A node to calculate args for default comfy node 'Pad Image For Outpainting'
    """
    ASPECT_RATIO_MAP = {
        "SDXL_1-1_square_1024x1024": (1024, 1024),
        "SDXL_4-3_landscape_1152x896": (1152, 896),
        "SDXL_3-2_landscape_1216x832": (1216, 832),
        "SDXL_16-9_landscape_1344x768": (1344, 768),
        "SDXL_21-9_landscape_1536x640": (1536, 640),
        "SDXL_3-4_portrait_896x1152": (896, 1152),
        "SDXL_5-8_portrait_832x1216": (832, 1216),
        "SDXL_9-16_portrait_768x1344": (768, 1344),
        "SDXL_9-21_portrait_640x1536": (640, 1536),
        "SD15_1-1_square_512x512": (512, 512),
        "SD15_2-3_portrait_512x768": (512, 768),
        "SD15_3-4_portrait_512x682": (512, 682),
        "SD15_3-2_landscape_768x512": (768, 512),
        "SD15_4-3_landscape_682x512": (682, 512),
        "SD15_16-9_cinema_910x512": (910, 512),
        "SD15_37-20_cinema_952x512": (952, 512),
        "SD15_2-1_cinema_1024x512": (1024, 512),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (list(s.ASPECT_RATIO_MAP.keys()), {"default": "SD1.5 - 1:1 square 512x512"}),
                "justification": (["top-left", "center", "bottom-right"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    NAMES = ("image", "new_width", "new_height", "padding_top", "padding_left")
    FUNCTION = "fit_and_calculate_padding"
    CATEGORY = "Eden 🌱/Image"

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