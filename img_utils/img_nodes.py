import torch
from PIL import Image
import sys, os, time
import torch
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
    

class Filepicker:
    def __init__(self):
        self.ci = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "pick_file"
    CATEGORY = "Eden 🌱"

    def pick_file(self, folder):
        files = os.listdir(folder)
        files = [os.path.join(folder, f) for f in files]
        files = [f for f in files if os.path.isfile(f)]

        random.shuffle(files)
        path = files[0]
        return (path,)


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
                "math_string": ("STRING",),
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
