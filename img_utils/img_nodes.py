import time
import os
import torch
import PIL.Image
from PIL import Image
import sys, os, time
import torch
import numpy as np
import random

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

        input_device = image.device
        input_dtype = image.dtype
        
        if image.max() > 1:
            rescaled = True
            # Normalize the image to a range of 0 to 1
            normalized_image = image / 255.0
        else:
            rescaled = False
            normalized_image = image

        # Apply the mathematical function defined by the string:
        # Prepare a lambda function from the math_string
        # Assuming 'x' represents the pixel value in math_string
        math_func = lambda x: eval(math_string)

        # Apply the mathematical function to each pixel
        transformed_image = torch.from_numpy(np.vectorize(math_func)(normalized_image.cpu().float().numpy()))

        # Clip values to ensure they are within 0 to 1
        transformed_image = torch.clamp(transformed_image, 0, 1)
        
        if rescaled:
            # Rescale back to 0-255 and convert to uint8
            output_image = transformed_image * 255

        # Convert to original device and dtype
        output_image = output_image.to(input_device, dtype=input_dtype)

        return (output_image,)

