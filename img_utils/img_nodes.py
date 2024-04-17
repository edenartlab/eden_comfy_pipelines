import torch
from PIL import Image
import sys, os, time
import torch
import cv2
import numpy as np
import random


###########################################################################

"""
Below is some dirty path hacks 
to make sure we can import comfyUI modules
"""

from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../comfy"))

# Import comfyUI modules:
from comfy.cli_args import args
import folder_paths

###########################################################################

import torch
from torch.cuda.amp import autocast

import psutil

def print_available_memory():
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / 1024 / 1024 / 1024:.2f} GB")


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

class SaveImageAdvanced:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
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
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()

        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        
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
                "type": self.type
            })

            if save_metadata_json and not args.disable_metadata:
                json_path = os.path.join(full_output_folder, file.replace(".png", ".json"))
                with open(json_path, "w") as f:
                    json.dump(metadata_dict, f, indent=4)                

            counter += 1

        return { "ui": { "images": results } }



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
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                    "folder": ("STRING", {"default": "."}),
                    "n_images": ("INT", {"default": 1, "min": 1, "max": 100}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 100000}),
                    "sort": ("BOOLEAN", {"default": False}),
                }
        }

    CATEGORY = "Eden 🌱"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"

    def load_image(self, folder, n_images, seed, sort):
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
        files = [f for f in files if os.path.isfile(f)]
        files = sorted([f for f in files if os.path.splitext(f)[1].lower() in self.img_extensions])

        random.seed(seed)
        random.shuffle(files)
        image_paths = files[:n_images]

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

        if len(output_images) > 1:
            output_images = get_uniformly_sized_crops(output_images)
            output_images = [torch.from_numpy(output_image)[None,] for output_image in output_images]
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = torch.from_numpy(output_images[0])[None,]

        return (output_image,)




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
    CATEGORY = "Eden 🌱"

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

        # plot the two histograms on the same graph:
        import matplotlib.pyplot as plt
        plt.plot(hist_input, label="Input")
        plt.plot(hist_output_avg, label="Output")
        plt.legend()
        plt.savefig("histograms.png")

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
    CATEGORY = "Eden 🌱"

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
    CATEGORY = "Eden 🌱"

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
    CATEGORY = "Eden 🌱"

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
    CATEGORY = "Eden 🌱"

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
    CATEGORY = "Eden 🌱"

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
    CATEGORY = "Eden 🌱"

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
